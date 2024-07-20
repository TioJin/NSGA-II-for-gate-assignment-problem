"""顺序遗传算法，交通运输工程学报"""

import math
import random
import copy
import sys
from typing import Optional
from dataclasses import dataclass

import numpy as np

from src.model.gate_graph import Flight, Gate, GateGraph
from src.airport import Airport
from src.constrain_check import static_constrain_checker
from src.property import attribute
from src.util import draw_graph, flight_utils
from src.util.time_machine import TimeMachine
from src.util.file import ExcelSave

# random.seed(41)

_near_gates = [gg for gg in Airport().get_gates() if Airport().is_near_gate(gg.gate.name) == True]
_remote_gates = [gg for gg in Airport().get_gates() if Airport().is_near_gate(gg.gate.name) == False]
# 首先将近机位放在前面，远机位放在后面，然后按照距离的距离排序。注意远机位的距离统一为200，比所有近机位的距离都大，因此不会影响近机位在前。
_gates = _near_gates + _remote_gates
_gates = sorted(_gates, key=lambda x: x.gate.distance_from_baggage_claim)
_gate_dict = {gg.gate.name : gg for gg in _gates}

def assign_gate_to_flight(fligths: list[Flight]):
    """按照传入的航班列表，按其顺序为每一架航班分配机位。
    """
    gate_dict = {gg.gate.name : list[Flight]() for gg in _gates}
    for f in fligths:
        for gg in _gates:
            if static_constrain_checker.check(gate=gg.gate, flight=f) == True:
                overlapping_flag = False
                for parked_f in gate_dict[gg.gate.name]:
                    if flight_utils.is_overlapping(f1=f, f2=parked_f) == True:
                        overlapping_flag = True
                        break # 当前时间段已经有航班停靠在该机位上，因此不能再停靠。
                if overlapping_flag == False:
                    gate_dict[gg.gate.name].append(f)
                    f.settle(gate_name=gg.gate.name)
                    break
    return gate_dict


@dataclass
class Schedule:
    gate_name: str # 机位名
    flight_schedule: list[str] # 机位对应的航班列表


class Chromosome:
    def __init__(self, gene: list[int]) -> None:
        self.gene = gene


@dataclass
class Individual:
    chromosome: Chromosome
    fitness: Optional[tuple] # could be multi-objective results
    rank: Optional[int]
    crowding_distance: Optional[float]


class NSGA2_AGAP:

    def __init__(self) -> None:
        self.flights = Airport().get_flights()
        self.gates = [gg.gate for gg in _gates]
        self.gate_dict = _gate_dict

    def generate_population(self, population_size: int) -> list[Individual]:
        individuals = []
        for _ in range(population_size):
            individual = self.make_an_individual()
            individual.fitness = self.calculate_fitness(individual=individual)
            individuals.append(individual)
        return individuals

    def calculate_fitness(self, individual: Individual) -> float:
        # 靠桥率计算。（靠桥率加负号是为了方便遗传算法的支配计算，加符号之后，靠桥率与冲突数量一样都是朝着最小的方向）
        parked_flights = convert_individual_to_fligths(individual=individual)
        gate_usage_rate = Airport().get_boarding_bridge_usage_rate(parked_flights)
        # 旅客行走距离，未下机的旅客数量。
        # total_passenger_distance, disembarking_passenger_count = flight_utils.calculate_total_passenger_distance(flights=parked_flights, gates=self.gates)
        size_fit = 0
        for f in parked_flights:
            if f.get_gate() is not None:
                size_fit += f.size / self.gate_dict[f.get_gate()].gate.size
        # 不考虑动态约束冲突，因此无需构图来判断软性的动态约束。静态约束是作为硬性约束，在初始种群生成时就考虑进去了。
        # 空闲时间方差计算
        # variance = flight_utils.calculate_idle_time_variance(flight_utils.to_gates_flights_dict(parked_flights))
        return (-gate_usage_rate, -size_fit)

    def crossover(self, parent1: Individual, parent2: Individual, crossover_rate:float) -> Individual:
        if parent1 is None or parent2 is None:
            raise(ValueError("参数传入错误，传入值为None"))
        if len(parent1.chromosome.gene) != len(self.flights) or len(parent2.chromosome.gene) != len(self.flights):
            raise(ValueError("个体的基因长度与航班数量不一致"))

        child1 = copy_individual(parent1)
        child2 = copy_individual(parent2)
        if random.random() <= crossover_rate:
            chromosome_len = len(child1.chromosome.gene)
            (left, right) = random.sample(range(0, chromosome_len - 1), 2) # 两点交叉
            temp = child1.chromosome.gene[left:right]
            child1.chromosome.gene[left:right] = child2.chromosome.gene[left:right]
            child2.chromosome.gene[left:right] = temp
            
            fix_gene(child1.chromosome.gene)
            fix_gene(child2.chromosome.gene)

        if len(child1.chromosome.gene) != len(self.flights) or len(child2.chromosome.gene) != len(self.flights):
            raise(ValueError("个体的基因长度与航班数量不一致"))

        return child1, child2

    def mutate(self, chromosome: Chromosome, mutate_rate:float) -> Chromosome:
        two_points = random.sample(range(0, len(chromosome.gene)), 2)
        if random.random() <= mutate_rate:
            chromosome.gene[two_points[0]], chromosome.gene[two_points[1]] = chromosome.gene[two_points[1]], chromosome.gene[two_points[0]]
        return chromosome

    def make_an_individual(self) -> Individual:
        gene = [i for i in range(len(self.flights))]
        random.shuffle(gene)
        new_individual = Individual(chromosome=Chromosome(gene=gene), fitness=None, rank=None, crowding_distance=None)
        return new_individual


def get_fligt_dict() -> dict[str, Flight]:
    """创建航班名与航班对象的字典，并返回该字典。本方法返回的航班是未经过构图的，直接从Airport().get_flights()拿来的。
    """
    new_flights = Airport().get_flights()
    flight_dict = dict[str, Flight]()
    for f in new_flights:
        if f.name not in flight_dict:
            flight_dict[f.name] = f
        else:
            raise(ValueError(f"重复航班{f.name}"))
    return flight_dict


def copy_individual(old_individual: Individual) -> Individual:
    """深度拷贝个体
    """
    if old_individual is None or old_individual.chromosome is None:
        raise(ValueError("个体的值存在错误"))
    
    return copy.deepcopy(old_individual)


def fix_gene(gene: list[int]) -> None:
    """
    直接在传入的基因列表上进行修改，不返回新的基因。
    返回的列表不存在重复值。如有重复值则会报错。
    """

    if max(gene) >= len(gene):
        raise(ValueError("基因编码错误，存在超出范围的编码。"))

    # 因去除重复而留下来的空缺位置
    empty_index = []
    # 保存安排机位的航班，默认没有航班被分配。
    index_dict = {}
    # 保存未安排机位的航班，默认所有飞机都未被分配。
    unset_index = { i : i for i in range(len(gene)) }
    for i in range(len(gene)):
        if gene[i] not in index_dict:
            index_dict[gene[i]] = i
        else:
            empty_index.append(i)

        if gene[i] in unset_index:
            unset_index.pop(gene[i]) # 去除已经分配的飞机

    if len(unset_index) != len(empty_index) or len(index_dict) + len(unset_index) != len(gene):
        raise(ValueError("未分配飞机的数量与空缺位置数量不一致"))

    unset_index = list(unset_index.keys())
    random.shuffle(unset_index)

    for i, v in enumerate(empty_index):
        gene[v] = unset_index[i]

    if len(set(gene)) != len(gene):
        raise(ValueError("有飞机重复，导致部分飞机缺失。"))


def convert_individual_to_fligths(individual: Individual) -> list[Flight]:
    """将个体转为航班列表
    """
    flights = Airport().get_flights()
    # 按照个体内的基因索引排列航班。
    index_flights = [flights[i] for i in individual.chromosome.gene]
    assign_gate_to_flight(index_flights)
    return index_flights


def save_individual(individual: Individual, save_path:str, time_consumed):
    """
    Parameters
    ----------
    individual: Individual
        种群中的个体。
    save_path: str
        文件的存储路径，例如：C:/Users/surface/OneDrive/Airport/Code/
    """
    flights = convert_individual_to_fligths(individual=individual)
    obj_values = [0, 2] # 更改此值可以控制在甘特图上的显示不同的指标值。
    Airport().save_scheduled_flights(scheduled_flights=flights, save_path=save_path, gantt_title="顺序编码遗传算法", 
                                     additional_infor=[f"耗时: {time_consumed}s"], display_obj_value=obj_values)
    Airport().save_information(file_path=save_path)


