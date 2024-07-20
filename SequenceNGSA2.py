"""顺序遗传算法，交通运输工程学报"""

import functools
import os
import random
import time

from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import tqdm

from heuristics_algorithm import NSGA2_S
from heuristics_algorithm.NSGA2_S import NSGA2_AGAP, Individual
from src.util.file import ExcelSave, write_txt
from src.util.draw_graph import plot_scatter
from src.util.utils import measure_execution_time


# random.seed(42)
# 定义问题参数
POP_SIZE = 100 # 种群大小 # 300
GENERATIONS = 100  # 迭代次数 # 500
CROSSOVER_RATE = 0.8 # 交叉率 # 0.7
MUTATION_RATE = 0.1 # 突变率 # 0.1

print("遗传算法参数: ", POP_SIZE, GENERATIONS, CROSSOVER_RATE, MUTATION_RATE)

nsga2_agap = NSGA2_AGAP()

# 交叉操作：均与交叉
def crossover(parent1: Individual, parent2: Individual):
    return nsga2_agap.crossover(parent1=parent1, parent2=parent2, crossover_rate=CROSSOVER_RATE)


# 突变操作：随机变异
def mutate(individual: Individual):
    return nsga2_agap.mutate(chromosome=individual.chromosome, mutate_rate=MUTATION_RATE)


# 生成个体
def generate_individuals(population_size=0) -> list[Individual]:
    return nsga2_agap.generate_population(population_size=population_size)


# 计算适应度值（目标函数值）
def calculate_fitness(individual: Individual):
    return nsga2_agap.calculate_fitness(individual)


# 选择操作：锦标赛选择
def tournament_selection(population: list[Individual], num_of_parents:int):
    """
    Parameters
    ----------
    population: list[Individual]
        The population that we chose parents from.
    num_of_parents: int
        num of returned parents.

    Return
    ------
    parents:list[Individual]
    """
    selected_parents = []
    while len(selected_parents) < num_of_parents:
        tournament = random.sample(population, 2)
        # Perform non-dominated sorting
        highest_rank = min(tournament, key=lambda x: x.rank).rank
        first_stage_winners: list[Individual] = [p for p in tournament if p.rank == highest_rank]
        # Choose the front with the highest rank
        # If there are multiple individuals in the highest front, use crowding distance
        if len(first_stage_winners) > 1:
            # Choose the individual with the highest crowding distance
            final_winner = max(first_stage_winners, key=lambda x:x.crowding_distance)
            selected_parents.append(final_winner)
        elif len(first_stage_winners) == 1:
            selected_parents.extend(first_stage_winners)
        else:
            raise(ValueError("Reached the unreachable area."))
    return selected_parents


# 快速非支配排序
# 在这里划分等级及支配
def fast_non_dominated_sort(population: list[Individual]):
    """
    Return
    ------
    a list of fronts, where each front is itself a list containing the indices of individuals belonging to that front. 
    Each front contains individuals that are non-dominated by any other individual in the population.
    """
    fronts = []  # List to store fronts
    S = [[] for _ in range(len(population))]  # List to store dominated individuals for each individual
    n = [0 for _ in range(len(population))]  # List to store domination count for each individual
    fronts.append([])  # Initialize the first front
    
    # Iterate through each individual in the population
    for i in range(len(population)):
        # Initialize domination count and set of dominated individuals
        n[i] = 0
        S[i] = []
        
        # Compare individual with every other individual
        for j in range(len(population)):
            if i == j: # itself
                continue
            
            # Check dominance relation
            if dominates(population[i], population[j]):
                S[i].append(j)  # Individual i dominates individual j
            elif dominates(population[j], population[i]):
                n[i] += 1  # Individual j dominates individual i
        
        # If no individual dominates individual i, assign it to the first front
        if n[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # Initialize front index
    i = 0
    
    # Loop to create subsequent fronts
    while len(fronts[i]) > 0:
        Q = []  # Queue to store individuals for the next front
        
        # Iterate through individuals in the current front
        for p in fronts[i]:
            # Iterate through individuals dominated by p
            for q in S[p]:
                n[q] -= 1  # Decrease domination count
                if n[q] == 0:
                    population[q].rank = i + 1
                    Q.append(q)  # Add individual to the queue if it becomes non-dominated
        
        # Move to the next front
        i += 1
        fronts.append(Q)
    
    # Remove empty fronts
    fronts.pop() # last list must be empty
    
    return fronts


# 一个个体是否支配另一个个体
def dominates(x, y):
    # Function to check if individual x dominates individual y
    return all(xi <= yi for xi, yi in zip(x.fitness, y.fitness)) and any(xi < yi for xi, yi in zip(x.fitness, y.fitness))


# 拥挤度计算
def crowding_distance_assignment(population: list[Individual]):
    """Calculate crowding distance value of the same front population.

    Parameters
    ----------
    population: list[Individual]
        Individual objects of the same front population.
    """

    num_individuals = len(population)
    num_objectives = len(population[0].fitness)
    indics = [i for i in range(0, num_individuals)]
    
    # Calculate crowding distance for each objective
    for obj_index in range(num_objectives):
        # Sort individuals based on the current objective within the front
        sorted_front = sorted(indics, key=lambda x: population[x].fitness[obj_index])
        
        # Set infinite crowding distance for boundary individuals
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')
        distance_between_max_min = (population[sorted_front[-1]].fitness[obj_index] - population[sorted_front[0]].fitness[obj_index])
        # Calculate crowding distance for internal individuals
        for i in range(1, num_individuals - 1):
            distance_between_front_back = (population[sorted_front[i+1]].fitness[obj_index] - population[sorted_front[i-1]].fitness[obj_index])
            if distance_between_max_min == 0:
                if distance_between_front_back != 0:
                    raise(ValueError("最大区间为0，却有子区间不为0。"))
                population[sorted_front[i]].crowding_distance = 0.0
            else:
                population[sorted_front[i]].crowding_distance = distance_between_front_back / distance_between_max_min
    

def sort_rank_and_calculate_crowding(population: list[Individual]):
    fronts = fast_non_dominated_sort(population=population)
    for front in fronts:
        front_individuals = [population[i] for i in front]
        crowding_distance_assignment(population=front_individuals)
    for individual in population:
        if individual.crowding_distance == None or individual.rank == None:
            raise(ValueError(f"After sorting rank and calculate crowding distance, some individuals still don't have the values.\n" +
                             f"crowding_distance: {individual.crowding_distance}, rank: {individual.rank}"))
    return fronts


# NSGA-II 主函数
@measure_execution_time
def nsga2():
    population = generate_individuals(population_size=POP_SIZE)
    sort_rank_and_calculate_crowding(population=population)
    best_indiv: Individual = None

    iteration_time = []
    for iteration in tqdm.tqdm(range(GENERATIONS), desc='NSGA2 Processing'):
        st = time.time()
        print(f"iteration: {iteration}")
        offspring_population = []
        #start = time.time()
        for _ in range(int(POP_SIZE / 2)):
            parent1, parent2 = tournament_selection(population=population, num_of_parents=2) # 选取一对父母
            children = crossover(parent1, parent2)
            #start = time.time()
            for child in children:
                mutate(child)
                child.fitness = calculate_fitness(individual=child)
            #end = time.time()
            #print(f"突变耗时：{end - start}s")
            offspring_population.extend(children)
        #end = time.time()
        #print(f"交叉变异耗时：{round(end - start, 2)}s")
        # 总群组合
        combined_population = population + offspring_population
        fronts = sort_rank_and_calculate_crowding(combined_population)
        new_population = []
        front_index = 0
        # Firstly, choose new individuals based on rank.
        # print(f"Num of F1 front: {len(fronts[front_index])}. Iteration: {iteration}")
        while len(new_population) + len(fronts[front_index]) <= POP_SIZE:
            individuals_of_the_same_rank = [combined_population[i] for i in fronts[front_index]]
            new_population.extend(individuals_of_the_same_rank)
            front_index += 1
        sorted_last_front = sorted(fronts[front_index], key=lambda x: combined_population[x].crowding_distance, reverse=True)
        new_population.extend([combined_population[i] for i in sorted_last_front[:POP_SIZE - len(new_population)]])
        population = new_population
        best_indiv, airbridge_usage, obj2 = best_indiv_of_every_iteration(population, best_indiv)

        print(f"最优个体靠桥率：{-airbridge_usage}, 机型大小匹配值{-obj2}")
        et = time.time()
        iteration_time.append(round(et - st, 2))

    return population, best_indiv, iteration_time


max_usage_of_every_iteration = []
conflict_num_of_every_iteration = []
def best_indiv_of_every_iteration(population: list[Individual], best_indiv: Optional[Individual] = None):

    def gate_usage_compare(indiv1: Individual, indiv2: Individual):
        for i in range(len(indiv1.fitness)):
            if indiv1.fitness[i] <= indiv2.fitness[i]:
                continue
            else: return 1
        return -1

    f1_front = [indiv for indiv in population if indiv.rank == 0]
    if best_indiv != None:
        f1_front.insert(0, best_indiv)
    f1_front = sorted(f1_front, key=functools.cmp_to_key(gate_usage_compare))
    max_usage_of_every_iteration.append(-f1_front[0].fitness[0])
    conflict_num_of_every_iteration.append(-f1_front[0].fitness[1])
    return f1_front[0], f1_front[0].fitness[0], f1_front[0].fitness[1]


if __name__ == "__main__":
    result, time_consumed = nsga2()
    final_population, best_indiv, iteration_time = result
    if len(final_population) > 0:
        num_of_object_funtcion = len(final_population[0].fitness)
    object_function_results = []
    object_function_results.extend([[] for _ in range(0, num_of_object_funtcion)])
    for individual in final_population:
        if len(individual.fitness) != num_of_object_funtcion:
            raise(ValueError(f"an individual's objective function went wrong. It has {len(individual.fitness)} objective funtcions, but the first individual has {num_of_object_funtcion}."))
        for i in range(0, len(individual.fitness)):
            object_function_results[i].append(individual.fitness[i])

    curr_path = os.path.dirname(os.path.abspath(__file__))
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    file_name = os.path.basename(__file__).replace(".py", "")
    result_path = f"{curr_path}/outputs-顺序遗传算法/{curr_time}/"

    NSGA2_S.save_individual(individual=best_indiv, save_path=result_path, time_consumed=time_consumed)

    write_txt(file_path=result_path, file_name="参数", 
              content=f"POP_SIZE: {POP_SIZE}，GENERATIONS: {GENERATIONS}, CROSSOVER_RATE: {CROSSOVER_RATE}, MUTATION_RATE: {MUTATION_RATE}")

    scatter_x = [-v for v in object_function_results[0]]
    scatter_y = [-v for v in object_function_results[1]]
    plot_scatter(x=scatter_x, y=scatter_y, xlable="靠桥率", ylable="机型大小匹配值", title="种群分布图", 
                 file_name="种群分布图", save_path=result_path)

    iteration = [iter for iter in range(GENERATIONS)]

    ExcelSave().save_arr_data(arr_data=[max_usage_of_every_iteration, conflict_num_of_every_iteration], save_path=result_path, file_name="迭代指标")
    ExcelSave().save_arr_data(arr_data=[iteration_time], save_path=result_path, file_name="迭代耗时")

    plot_scatter(x=iteration, y=max_usage_of_every_iteration, xlable="迭代轮次", ylable="靠桥率", title="靠桥率收敛图", 
                 file_name="靠桥率", save_path=result_path)
    
    plot_scatter(x=iteration, y=conflict_num_of_every_iteration, xlable="迭代轮次", ylable="机型大小匹配值", title="机型大小匹配值收敛图", 
                 file_name="机型大小匹配值", save_path=result_path)

