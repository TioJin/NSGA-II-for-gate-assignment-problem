# 下方用词中，飞机==航班。


import sys
import time
from typing_extensions import deprecated
import numpy as np
import functools
import copy
from typing import Any, Literal, Optional
from abc import ABC, abstractmethod
from collections import deque

from ..property import attribute
from ..util.time_machine import TimeMachine

from ..util.utils import measure_execution_time


"""
ArcBox和VexNode与机位分配无关，属于残余代码，无视就行。
"""

# 十字链表法存储图结构
# 弧结点
class ArcBox:

    def __init__(self, tail_vex, head_vex, head_link=None, tail_link=None, info=None, cross_pre_link=None):
        self.tail_vex:VexNode = tail_vex  # 父节点
        self.head_vex:VexNode = head_vex  # 子节点
        self.head_link:ArcBox = head_link  # 当前子节点的下一个父节点
        self.tail_link: ArcBox = tail_link  # 当前父节点的下一个子节点
        # 横向前继
        self.cross_pre_link: ArcBox = cross_pre_link  # A --> B, A ---> C, 这是两条有先后顺序的弧，是A所指向的所有节点，
                                  # B没有pre_link，因为其前面直接就是A节点，但C有，C的pre_link就是A指B的那条弧ArcBox。
                                  # tail_link充当后向指针，pre_link充当前向指针。pre_link为空时，说明当前节点是头节点；tail_link为空时，说明当前节点是尾节点。
        # 纵向前继
        self.vertical_pre_link: ArcBox = None

        self.info = info  # 额外信息补充，不过目前并没有什么额外信息需要使用到该变量。

    def __str__(self) -> str:
        returned_str = f"tail_vex: {self.tail_vex.name}, head_vex: {self.head_vex.name}"
        return returned_str


# 图中的顶点
class VexNode():

    def __init__(self, name: str, first_in: ArcBox=None, first_out: ArcBox=None):
        self.name = name
        self.first_in: ArcBox = first_in
        self.first_out: ArcBox = first_out

    # 比较节点名字
    def __eq__(self, vex: object) -> bool:
        if isinstance(vex, VexNode):
            return self.name == vex.name
        return False

    def clear_arc(self):
        """清除节点的弧。
        """
        self.first_in = None
        self.first_out = None
    

# 机位
class Gate(VexNode):

    def __init__(self, name: str, size: int, airline: list, inter_or_domestic: int, passenger_or_cargo: int, 
                 open_time: int = 0, stand_type: Literal["近机位", "远机位"] = None, distance_from_baggage_claim: float = 0):
        super().__init__(name=name)
        self.open_time = open_time  # 机位开放时间
        # 机位-机型机位容量
        self.size = size
        # 机位-航司（可突破，但尽量不要出现）
        self.airline = airline
        # 机位-属性（国际国内）（可突破，但尽量不要出现）
        self.inter_or_domestic = inter_or_domestic
        # 机位-任务（客货运）（可突破，但尽量不要出现）
        self.passenger_or_cargo = passenger_or_cargo
        # 机位类型
        self.stand_type = stand_type
        if stand_type != None:
            if stand_type != "近机位" and stand_type != "远机位":
                raise(ValueError(f"机位类型输入错误：{stand_type}"))
        self.distance_from_baggage_claim = distance_from_baggage_claim  # 机位距离航站楼的距离

    @staticmethod
    def forge_gate(name: str, cls = None, size = None, airlines = None, inter_or_domestic = None, passenger_or_cargo = None):
        """
            该方法产生虚拟机位, 各属性随机取值。

            特殊机位的返回特殊机位实例，且属性取值可能有所不同，具体看特殊机位类的该方法实现。
        """
        size = attribute.attr_random_size() if size is None else size
        airlines = attribute.attr_airline(values=True, num=-1) if airlines is None else airlines 
        inter_or_domestic = attribute.attr_domestic_and_international() if inter_or_domestic is None else inter_or_domestic
        passenger_or_cargo = attribute.attr_passenger_and_cargo() if passenger_or_cargo is None else passenger_or_cargo
        cls = Gate if cls is None else cls
        # g = cls(name=name, size=attribute.get_maxsize(), airline=attribute.attr_airline(values=True, num=-1),
        #         inter_or_domestic=attribute.attr_domestic_and_international(), passenger_or_cargo=attribute.attr_passenger_and_cargo())  # “全能”机位，可容纳任何属性的飞机。
        # return g
        return cls(name=name,
                    size=size,
                    airline=airlines,
                    inter_or_domestic=inter_or_domestic,
                    passenger_or_cargo=passenger_or_cargo)

    def constrain_check(self, flight) -> bool:
        """机位的动态约束检查。
        通过约束检查返回True，反之False。
        """
        # 特殊机位的约束检查接口，例如组合机位、父子机位等。
        # 普通机位此方法默认返回True。
        return True


# 航班
class Flight(VexNode):
    
    # 航班号并非唯一值，如果读取多天的数据，那么航班号就可能发生重复；当航班因拖曳而产生多段分割时，除首段使用进港航班号外，余下段都使用出港航班号，这同样导致航班号重复问题。
    # 而self.name的假设条件就是唯一的，因此，为确保self.name唯一，self.name不由外部赋值，而由内部确定，其作用相当于id。
    id = 0
    
    def __init__(self, name: str, size: int, inter_or_domestic: int, passenger_or_cargo: int, 
                 arrival_time: int, departure_time: int, airline: int, aircraft_Reg_No:str = None, custom_name: bool = False,
                 passenger_num: int = None):
        # 当copy_flag = False代表创建新的对象，反之代表复制现有对象。
        if custom_name is False:
            self.set_name(name)
        else:
            self.name = name
        super().__init__(name=self.name)
        # 机位-机型机位容量
        self.size = size if type(size) == int else attribute.attr_size(size)
        # 机位-属性（国际国内）（可突破，但尽量不要出现）
        self.inter_or_domestic = inter_or_domestic
        # 机位-任务（客货运）（可突破，但尽量不要出现）
        self.passenger_or_cargo = passenger_or_cargo
        # 飞机所属航空公司(如果为空就随便给一个)
        self.airline = airline
        # 进港时间
        self.arrival_time = arrival_time
        # 离港时间
        self.departure_time = departure_time
        # 经调度算法所决定的机位
        self._settled = None
        # 实际进离港时间，用于后期考虑延误情况
        self.actual_arrival_time = arrival_time
        self.actual_departure_time = departure_time
        # 航班所使用航空器的注册编号
        self.aircraft_Reg_No = aircraft_Reg_No
        # 乘客数量
        self.passenger_num = passenger_num

    def set_name(self, name):
        Flight.id += 1
        self.flight_No = name # 航班号
        self.name = str(Flight.id) + "-" + name

    def settle(self, gate_name: str):
        """将飞机停在某机位
        """
        if gate_name is None:
            raise(ValueError("机位名称为空"))
        self._settled = gate_name

    def reset(self):
        """重置机位为空。即该航班并未被调度至机位。
        """
        self._settled = None

    def get_gate(self):
        """
            返回飞机所停机位(经由调度算法确定下来的)。

            Returns
            -------
            如果该航班已被调度，则返回机位名字，未被调度，返回None。
        """
        return self._settled

    @staticmethod
    def forge_flight(name: str, aircraft_Reg_No: str=None, at: int=None, dt: int=None, size: str=None, inter_or_domestic: int=None, passenger_or_cargo: int=None):
        """
            该方法产生虚拟航班, 不赋予值的属性将保留空值或随机取值。

            params:
                at: 时间戳格式的到达时间
                dt: 时间戳格式的离场时间
        """
        size = attribute.attr_random_size() if size is None else attribute.attr_size(size)
        inter_or_domestic = attribute.attr_random_domes_inter() if inter_or_domestic is None else inter_or_domestic
        passenger_or_cargo = attribute.attr_random_passenger_cargo() if passenger_or_cargo is None else passenger_or_cargo
        time_slot = TimeMachine.get_time_slot(min_interval=50, max_interval=240)
        arrival_time = time_slot[0] if at is None else at
        departure_time = time_slot[1] if dt is None else dt
        airline = attribute.attr_airline()
        return Flight(name=name,
                      size=size,
                      inter_or_domestic=inter_or_domestic,
                      passenger_or_cargo=passenger_or_cargo,
                      arrival_time=arrival_time,
                      departure_time=departure_time,
                      airline=airline,
                      aircraft_Reg_No=aircraft_Reg_No)
    
    def __str__(self) -> str:
        infor = f"id:{self.name} flight_No:{self.flight_No} AT:{TimeMachine.time_stamp_to_time(self.arrival_time)} DT:{TimeMachine.time_stamp_to_time(self.departure_time)} size:{self.size} gate:{self._settled}"
        return infor


class GateGraph:
    """
    没有的类，以前的代码残余。删的话影响太多了，就不删了。（屎山代码
    """

    def __init__(self, gate: Gate):
        """绑定机位

        Parameters
        ----------
        gate: Gate
            机位图与机位是一体的，因此在初始化机位图时，需传入一个机位进行绑定。

        """
        self.gate = gate


def sort_flights(flights: list[Flight]) -> list[Flight]:
    """
        飞机按照进港时间从早到晚排序。
    """
    return sorted(flights, key=lambda x: x.arrival_time)


def get_safe_gap(flight: Flight) -> int:
    """返回8分钟安全时间间隔。
    """
    if flight is None:
        raise(ValueError("输入为空"))
    return TimeMachine.get_minute(8)

