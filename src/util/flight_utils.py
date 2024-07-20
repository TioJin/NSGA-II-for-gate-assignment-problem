import functools
from typing import Literal, Optional

import numpy as np

from ..property import attribute
from .time_machine import TimeMachine
from ..model.gate_graph import Flight, Gate, GateGraph


def sort_flights(flights: list[Flight], flag: Literal[0, 1] = 0) -> list[Flight]:
    """对航班列表进行升序排序，并将排好序的航班返回。

    Parameters
    ----------
    flag: 0 | 1
        0: 按照进港时间升序。
        1: 按照出港时间肾虚。
    """
    if flag == 0:
        flights = sorted(flights, key=lambda x: x.arrival_time)
    elif flag == 1:
        flights = sorted(flights, key=lambda x: x.departure_time)
    else:
        raise(ValueError(f"参数输入错误：{flag}"))
    return flights


def to_gates_flights_dict(flights: list[Flight]):
    """给定航班列表，转为机位-航班字典。

    Returns
    -------
    gates_flights_dict: dict[str, list[Flight]]
        机位-航班字典。
    unparked_flights: list[Flight]
        未停靠机位的航班列表。
    """
    gates_flights_dict = dict[str, list[Flight]]()
    unparked_flights = list[Flight]()
    for f in flights:
        gate_name = f.get_gate()
        if gate_name == None: 
            unparked_flights.append(f)
        if gate_name not in gates_flights_dict:
            gates_flights_dict[gate_name] = []
        gates_flights_dict[gate_name].append(f)
    return gates_flights_dict, unparked_flights


def min_max_time(flights: list[Flight]):
    """给定航班列表，返回最早到达时间和最晚离场时间。
    """
    min_arrival_time = sort_flights(flights=flights)[0].arrival_time
    max_departure_time = sort_flights(flights=flights, flag=1)[-1].departure_time
    return min_arrival_time, max_departure_time


def pack_flights_to_arr(flights: list[Flight], gates: list[Gate | GateGraph]):
    """将航班数据存至二维数组中。参数gates用以确定航班所停航班的类型（近机位、远机位），因此gates必须包含flights所停靠的所有机位。
    """
    if type(gates[0]) != Gate and type(gates[0]) != GateGraph:
        raise(ValueError(f"输入的机位参数类型错误：{type[gates[0]]}"))

    if type(gates[0]) == GateGraph:
        gates = [gg.gate for gg in gates] # 从机位图中抽出机位。
    gate_dict = dict[str, Gate]()
    for gate in gates:
        if gate.name in gate_dict:
            raise(ValueError(f"存在相同机位，重复机位的名是：{gate.name}"))
        gate_dict[gate.name] = gate

    arr_data = []
    sheet_titles = ["机号", "航班号", "机型大小", "乘客数量", "国内国际", "客货运", "所属航司", "进港时间", "离港时间",
                    "停靠机位", "停靠机位类型", "停机位大小", "机位距行李处距离"]
    arr_data.append(sheet_titles)
    for f in flights:
        f: Flight = f
        gate = gate_dict[f.get_gate()] if f.get_gate() != None else None
        stand_type = gate.stand_type if gate != None else ""
        gate_size = attribute.size_to_str(gate.size) if gate != None else ""
        distance_from_baggage_claim = gate.distance_from_baggage_claim if gate != None else ""
        f_information = [f.aircraft_Reg_No, f.flight_No, attribute.size_to_str(f.size),
                         f.passenger_num, f.inter_or_domestic, f.passenger_or_cargo, f.airline, 
                         TimeMachine.time_stamp_to_time(f.arrival_time), 
                         TimeMachine.time_stamp_to_time(f.departure_time),
                         f.get_gate(), stand_type, gate_size, distance_from_baggage_claim]
        arr_data.append(f_information)
    return arr_data


def calculate_idle_time_variance(gate_flight_dict: dict[str, list[Flight]]) -> float:
    """
    计算排班计划的机位空闲时间方差
    """
    interval_time_list = []
    flight_list = []
    for _, values in gate_flight_dict.items():
        flights = sort_flights(values)
        for i, f in enumerate(flights):
            flight_list.append(f.name)
            if i == 0: # 不计算机位上的第一个航班的间隔时间（它前面没有航班啊）
                continue
            interval_time_list.append(f.arrival_time - flights[i - 1].departure_time)
    if len(flight_list) != len(tuple(flight_list)):
        raise(ValueError("航班数据错误，存在一个航班在多个机位，不是调度后的排班表。"))
    variance = np.var(interval_time_list)
    return round(variance, 2)


def calculate_total_passenger_distance(flights: list[Flight], gates: list[Gate]):
    """
    Calculates the total distance traveled by passengers in planes.

    Parameters:
    -----------
    flights (list[Flight]): 
        A list of Flight objects representing the flights.
    gates (list[Gate]): 
        A list of Gate objects representing the gates that those flights parked.

    Returns:
    --------
    total_distance: int,
        The total distance to baggage claim traveled by passengers.
    disembarking_passenger_count: int,
        The number of passengers of flights having no stand to park.
    """
    
    gate_dict = dict[str, Gate]()
    for g in gates:
        gate_dict[g.name] = g
    total_distance = 0.0
    disembarking_passenger_count: int = 0 # 未下机的旅客数量
    for f in flights:
        if f.get_gate() != None:
            gate = gate_dict[f.get_gate()]
            distance = gate.distance_from_baggage_claim * f.passenger_num # “旅客数量”乘以“机位至行李处距离”
            total_distance += distance
        else:
            disembarking_passenger_count += f.passenger_num
    total_distance += disembarking_passenger_count * 200 # 未下机的旅客 * 200，加在总行走距离上，作为惩罚。
    return round(total_distance, 2), disembarking_passenger_count


def is_overlapping(f1: Flight, f2: Flight, safe_gap: Optional[int] = None, external_time: int = 0) -> bool:
    """
    判断两个航班的停留时间是否重叠（考虑机位安全时间间隔）。重叠返回True，未重叠返回False。

    参数：
        safe_gap: 自定义安全间隔时间。取值为None或int，如果取值为None，则安全间隔时间取决于后一架航班。
        external_time: 额外的自定义间隔时间，会在加安全间隔时间上。

        时间为时间戳格式，即以秒为单位。
    """
    if f1 is None or f2 is None:
        raise(ValueError("传入参数为空"))
    
    flights = [f1, f2]
    flights = sorted(flights, key=lambda x: x.arrival_time)
    for i in range(len(flights) - 1):
        # 安全间隔时间取决于前一架航班
        safe_gap = get_safe_gap(flights[i]) if safe_gap is None else safe_gap 
        if flights[i].departure_time + safe_gap + external_time > flights[i+1].arrival_time:
            return True # 存在冲突
    
    return False # 不存在冲突


def check_flights_overlapping(flights: list[Flight]) -> bool:
    """
    判断航班列表中是否存在任意两架航班有时间重叠冲突。存在时间重叠则返回True，反之False。
    """
    flights = sort_flights(flights)
    for i in range(len(flights) - 1):
        safe_gap = get_safe_gap(flights[i]) # 安全间隔时间取决于前一架航班
        if flights[i].departure_time + safe_gap > flights[i+1].arrival_time:
            return True # 存在冲突
        
    return False # 不存在冲突


def get_safe_gap(flight: Flight) -> int:
    """返回安全间隔时间，单位为分钟。安全间隔时间取决于飞机的机型等级。
    """
    if flight is None:
        raise(ValueError())
    return TimeMachine.get_minute(8)


def copy_flight(flight: Flight) -> Flight:
    """
    复制航班对象。
    """
    new_flight = Flight(name=flight.name, size=flight.size, inter_or_domestic=flight.inter_or_domestic, passenger_or_cargo=flight.passenger_or_cargo, 
                        arrival_time=flight.arrival_time, departure_time=flight.departure_time, airline=flight.airline, 
                        aircraft_Reg_No=flight.aircraft_Reg_No,passenger_num=flight.passenger_num, custom_name=True)
    new_flight.flight_No = flight.flight_No
    if flight.get_gate() != None:
        new_flight.settle(flight.get_gate())
    return new_flight


