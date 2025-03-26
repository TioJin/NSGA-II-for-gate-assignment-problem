import copy
import random
import sys
from typing import Literal, Optional
from datetime import datetime

import functools

from .property import attribute
from . import airport_data
from src.model.gate_graph import Gate, Flight, GateGraph

from .util.file import ExcelSave, write_txt
from .util.draw_graph import draw_gantt
from .util.time_machine import TimeMachine
from .util import flight_utils

DATA_NUM = 100000
NEAR_GATE_NUM = 20 # 近机位数目
REMOTE_GATE_NUM = 20 # 远机位数目

FLIGHT_NUM = 150 # 航班数目。

NEAR_GATE = "近机位"
REMOTE_GATE = "远机位"

print(f"airport参数: ", "近机位数量:{NEAR_GATE_NUM}", "远机位数量:{REMOTE_GATE_NUM}")


def _add_distance_info(func):
    """
    为机位列表添加距离信息。

    Parameters
    ----------
    gates: list[Gate]
        机位列表。

    distance: float
        距离值。

    Returns
    -------
    list[Gate]
        添加了距离信息的机位列表。
    """
    def wrapper(*args, **kwargs):
        near_gates: list[GateGraph] = None
        remote_gates: list[GateGraph] = None
        near_gates, remote_gates = func(*args, **kwargs)
        distance = 0
        for gg in near_gates:
            gg.gate.distance_from_baggage_claim = distance
            distance += 1
        for gg in remote_gates:
            gg.gate.distance_from_baggage_claim = 200 # 远机位统一距离
        return near_gates, remote_gates
    return wrapper


class Airport:
    """
        鉴于该类的任务性质，将该类设计为单例模式。

        负责任务：
            1、划分机场停机位区域，近机坪，远机坪等。\n
            2、匹配度计算。（停在什么类型的机位、停在远机位还是近机位等等）\n
            3、机位数据生成。\n
            4、场面约束。\n
    """

    _instance = None
    _once_init = False

    # 单例模式
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self) -> None:
        # 配合单例模式，数据仅需生成一次。
        if Airport._once_init == False:
            print("|Airport -> __init__()| 机场实例化*************************")
            # 存放近机位的列表
            self.near_gates = list[str]() # 元素类型为字符串类型，存储机位名。
            # 存放远机位的列表
            self.remote_gates = list[str]()
            # 随机生成机位与航班数据
            self.gates = list[GateGraph]()
            self._generate_data() 
            print(f"|Airport -> __init__()| 所使用的：近机位{len(self.near_gates)}个 + 远机位{len(self.remote_gates)} = {len(self.gates)}个")
            print(f"|Airport -> __init__()| 航班数量共：{FLIGHT_NUM}")
            Airport._once_init = True

    def _as_near_gates(self, gates: list[GateGraph]):
        """
            将某些机位划分为近机位
        """
        for gg in gates:
            gg.gate.stand_type = NEAR_GATE # 这是标识一个机位是远机位/近机位，方便后面保存excel文件时标识该机位。
            gg.gate.name = gg.gate.name.replace("OG", "gate")
            self.near_gates.append(gg.gate.name)
        
    def _as_remote_gates(self, gates: list[GateGraph]):
        """
            将某些机位划分为远机位
        """
        for gg in gates:
            gg.gate.stand_type = REMOTE_GATE
            gg.gate.name = gg.gate.name.replace("OG", "apron")
            self.remote_gates.append(gg.gate.name)

    @_add_distance_info
    def _generate_data(self, gate_start_id = 0, flight_start_id = 0):
        """
            生成虚拟机位与航班数据
        """
        near_gates = [GateGraph(gate) for gate in airport_data.get_gates(gate_num=NEAR_GATE_NUM, start_id=gate_start_id)]
        self._as_near_gates(near_gates)
        remote_gates = [GateGraph(gate) for gate in airport_data.get_gates(gate_num=REMOTE_GATE_NUM, start_id=gate_start_id + NEAR_GATE_NUM)]
        self._as_remote_gates(remote_gates)
        self.gates = near_gates + remote_gates
        self.gate_graph_dict = {gg.gate.name: gg for gg in self.gates}
        
        self.flights = airport_data.get_flights(FLIGHT_NUM, start_id=flight_start_id)
        self.flights_dict = {f.name: f for f in self.flights}

        print(f"|airport -> _generate_data()|: 近机位数量{len(self.near_gates)}，远机位数量{len(self.remote_gates)}")
        print(f"|airport -> _generate_data()|: 航班数量:{len(self.flights)}, 机位数量:{len(self.gates)}")

        return near_gates, remote_gates

    def get_gates(self) -> list[GateGraph]:
        """Return a new gate graph list.
        """
        return copy.deepcopy(self.gates)
    
    def get_flights(self) -> list[Flight]:
        """Return a new flight list.
        """
        return copy.deepcopy(self.flights)

    def get_boarding_bridge_usage_rate(self, flights: list[Flight]):
        """
        给定已停靠航班集合，返回靠桥率。（直接将拿出去的航班再传回来就行）

        Parameter:
        ---------
        flights: 已停靠航班集合。
        """
        num_of_flights_using_bridge = 0
        for f in flights:
            if f.name not in self.flights_dict: raise(ValueError(f"id: {f.name} flight_No: {f.flight_No} 不存在于当前的航班数据。属性可能发送了错误赋值。"))
            if f.get_gate() in self.near_gates:
                num_of_flights_using_bridge += 1
        rate = (num_of_flights_using_bridge / len(self.flights)) * 100 # 这里除以len(self.flights)，意味着传进来的flights参数未包括所有航班也能计算正确的靠桥率。
        return round(rate, 2)

    def is_near_gate(self, gate_name: str):
        """
            判断某机位是否为近机位。为近机位返回True，反之False。
        """
        if gate_name in self.near_gates:
            return True
        return False

    def save_information(self, file_path:str):
        """将机场的相关设置保存至指定文件夹，包括参数、当前航班数据的人工机位方案等。

        Parameters
        ----------
        file_path: str
            文件的保存路径。例如：C:/Users/24246/OneDrive/Airport/
        """ 

        # 保存设置的机场数据信息
        write_txt(file_path=file_path, file_name="机场数据信息", 
                content=f"\n{datetime.now()}\n "
                + f"DATA_NUM: {DATA_NUM}, NEAR_GATE_NUM: {NEAR_GATE_NUM}, REMOTE_GATE_NUM: {REMOTE_GATE_NUM}, FLIGHT_NUM: {FLIGHT_NUM}")

    def save_scheduled_flights(self, scheduled_flights: list[Flight], save_path: str, gantt_title: str, additional_infor: list = [],
                               manual: bool = False, display_obj_value: list[int] = [0]):
        """保存已调度完成的航班列表，即保存调度结果。

        Parameters
        ----------
        scheduled_flights: 经过调度后航班列表，列表内的航班的gate属性有值，则认为停靠在对应机位上，未设置就认为没有机位可停靠。
        save_path: 航班信息保存地址。
        gantt_title: 甘特图的标题，同时作为甘特图的文件名。
        additional_infor: 添加在甘特图上的额外信息，例如最大奖励值。
        manual: 是否为人工调度。
        display_obj_value: 选择需要计算并显示在甘特图的指标值。
            可选择值：[0, 1, 2, 3, 4]，0: 靠桥率，1: 空闲时间方差，2: 机型大小匹配值，3: 旅客总步行距离，4: 未下机的旅客数量。
        """

        display_obj_value = copy.copy(display_obj_value)

        used_gates = [gg.gate for gg in self.gates]

        # 人工调度结果所使用的机位与算法调度所使用的机位不一致（当然，是否一致还是取决于是否使用了文件中的所有机位：NEAR_GATE_NUM = ？ REMOTE_GATE_NUM = ?）。
        used_gate_names = [gate.name for gate in used_gates]
        f_gate = [] # 停在近机位上的航班
        f_apron = [] # 停在远机位上的航班
        f_unparked = [] # 未安排机位的航班
        attr_size = 0 # 机型大小匹配
        for f in scheduled_flights:
            parked_gate = f.get_gate()
            if parked_gate == None: 
                f_unparked.append(f)
                continue
            if parked_gate not in used_gate_names: 
                raise(ValueError(f"航班{f.flight_No}停靠在了不存在的机位：{f.get_gate()}"))
            # 计算机型大小匹配
            if f.get_gate() != None:
                attr_size += f.size / self.gate_graph_dict[f.get_gate()].gate.size
            if parked_gate in self.near_gates:
                f_gate.append(f)
            else:  
                f_apron.append(f)
        f_parked = f_gate + f_apron # 近机位航班在前，远机位航班在后。这一步对于最后显示的甘特图有影响，使得近机位和远机位区分开来。
        
        # 机位-航班列表字典
        gate_flights_dict, _ = flight_utils.to_gates_flights_dict(f_parked)
        # 计算靠桥率
        airbridge_usage = self.get_boarding_bridge_usage_rate(scheduled_flights)
        # 计算空闲时间方差
        varience = flight_utils.calculate_idle_time_variance(gate_flights_dict)
        # 计算总的旅客行走距离（离行李处的距离）、未安排机位的航班的旅客总数量
        total_passenger_distance, disembarking_passenger_count = flight_utils.calculate_total_passenger_distance(flights=scheduled_flights, 
                                                                                                                 gates=used_gates)
        
        obj_values = [f"靠桥率: {airbridge_usage}%", f"空闲时间方差: {varience}", f"机型大小匹配值: {attr_size}", 
                      f"旅客总步行距离: {total_passenger_distance}", f"未下机的旅客数量: {disembarking_passenger_count}"]

        used_gate_names = list(gate_flights_dict.keys())
        near_gate_num = 0
        remote_gate_num = 0
        # 统计使用的近远机位数目
        for g_name in used_gate_names:
            if self.is_near_gate(g_name):
                near_gate_num += 1
            else:
                remote_gate_num += 1

        infor = []
        # 添加外部信息
        infor.extend(additional_infor)
        # 把靠桥率放在第一位显示
        if 0 in display_obj_value:
            infor.insert(0, obj_values[0]) 
            display_obj_value.remove(0)
        # 添加目标函数信息
        for dov in display_obj_value:
            infor.append(obj_values[dov])

        # 为甘特图填充信息，使得左右两部分的信息区分开来。
        if len(infor) < 4:
            for _ in range(4 - len(infor)):
                infor.append("")
        
        # 添加一些统计信息
        infor.extend([f"停靠航班数量:{len(f_gate) + len(f_apron)}", f"未停靠航班数量:{len(f_unparked)}",
                 f"近机位数量:{near_gate_num}，远机位数量:{remote_gate_num}", 
                 f"近机位航班数量：{len(f_gate)}，远机位航班数量：{len(f_apron)}"])

        # 保存甘特图
        draw_gantt(gates_flights_dict=gate_flights_dict, title=f"{gantt_title}", save_path=save_path, 
                   additional_infor=infor, display_name=True)
        draw_gantt(gates_flights_dict=gate_flights_dict, title=f"{gantt_title}", save_path=save_path, 
                   additional_infor=infor, display_name=False)
        ExcelSave().save_arr_data(arr_data=flight_utils.pack_flights_to_arr(flights=scheduled_flights, gates=self.gates), 
                                  save_path=save_path, file_name=gantt_title + " 生成航班表")

