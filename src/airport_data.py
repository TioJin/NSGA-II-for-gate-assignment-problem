"""
    机位与航班数据的生成与处理模块
"""

from dataclasses import dataclass
import datetime
import random
import sys
import functools
from typing import Optional
import pandas as pd
import numpy as np
import copy
import openpyxl

from itertools import islice

from src.util import flight_utils

from .property import attribute
from .model.gate_graph import Gate, Flight
from .util.time_machine import TimeMachine
from .util.file import read_excel


def __generate_gates(cls, num, start_id=0, name="", size=None, inter_or_domestic=None, passenger_or_cargo=None):
    gate_list = list[Gate]()
    for _ in range(num):
        g = Gate.forge_gate(name = f"g{start_id}{name}", cls=cls, size=size, inter_or_domestic=inter_or_domestic, passenger_or_cargo=passenger_or_cargo)
        start_id += 1
        gate_list.append(g)
    return gate_list, start_id


def get_ordinary_gates(num, start_id=0, size=None):
    return __generate_gates(cls=Gate, num=num, start_id=start_id, name="-OG", size=size)


def get_gates(gate_num = 0, start_id = 0):
    """
    生成指定数量的各类机位。

    Parameters
    ----------
    gate_num: int
        普通机位的数量。

    Returns
    -------
    out: list[Gate]
        普通机位列表
    """

    if gate_num > 300 or gate_num < 1:
        raise(ValueError("机位数量过于苛刻"))

    e = int(gate_num * 0.2) # F属性机位的数量
    c = gate_num - e # C属性机位的数量

    # 属性列表
    attr_size = [attribute.attr_size("C")] * c + [attribute.attr_size("E")] * e
    ordinary_gates_list = []
    for _ in range(gate_num):
        gate, start_id = get_ordinary_gates(num=1, start_id=start_id, size=attr_size.pop())
        ordinary_gates_list.extend(gate)
    
    return ordinary_gates_list


def get_flights(num, start_id=0):
    flights = list[Flight]()
    # 各大小占比，c型机占百分之80以上。
    a = int(num * 0.02)
    b = int(num * 0.05)
    d = int(num * 0.03)
    e = int(num * 0.1)
    c = num - a - b - d - e 
    attr_size = ["A"] * a + ["B"] * b + ["C"] * c + ["D"] * d + ["E"] * e

    for _ in range(num):
        f = Flight.forge_flight(name=f"F{start_id}", size=attr_size.pop())
        f.passenger_num = _get_passenger_count(f)
        flights.append(f)
        start_id += 1
    return flights


# 航班排序的对比方法
def arriving_compare(tuple1, tuple2):
    """
        飞机按照进港时间从早到晚排序所需的对比方法。
    """
    if tuple1[0].arrival_time > tuple2[0].arrival_time:
        return 1
    else:
        return -1


# 时间段检查
def _date_check(left, right, f:Flight):
    # 限制进出港时间
    if f.arrival_time >= left and f.departure_time <= right:
        return True
    
    # 仅限制进港时间
    # if f.arrival_time >= left and f.arrival_time <= right:
    #     return True
    return False


# 检查航班的停场时间是否超过某一阈值
def _check_parking_time(f:Flight, threshold):
    if f.departure_time - f.arrival_time > threshold:
        return False
    return True


def _get_passenger_count(flight:Flight) -> int:
    """
    根据航班大小属性，返回乘客数量。
    """
    if flight.size == attribute.attr_size("A"):
        return random.randint(5, 10)
    elif flight.size == attribute.attr_size("B"):
        return random.randint(15, 20)
    elif flight.size == attribute.attr_size("C"):
        return random.randint(25, 30)
    elif flight.size == attribute.attr_size("D"):
        return random.randint(35, 40)
    elif flight.size == attribute.attr_size("E"):
        return random.randint(55, 85)
    else:
        raise(ValueError(f"飞机大小属性值错误，出现该值：{flight.size}"))


def _add_passenger_count(func):
    """为航班添加乘客数量属性。
    """
    def wrapper(*args, **kwargs):
        flights: list[Flight] = None
        result = func(*args, **kwargs)
        flights = result[0]
        for flight in flights:
            flight.passenger_num = _get_passenger_count(flight)  # Replace get_passenger_count with the actual function to calculate passenger count
        return result
    return wrapper


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
        near_gates: list[Gate] = None
        remote_gates: list[Gate] = None
        near_gates, remote_gates = func(*args, **kwargs)
        distance = 0
        for gate in near_gates:
            gate.distance_from_baggage_claim = distance
            distance += 1
        for gate in remote_gates:
            gate.distance_from_baggage_claim = len(near_gates) * 2
        return near_gates, remote_gates
    return wrapper


class RealAirportData:
    """读取航班表
    """
    
    _default_file_path = r"C:\Users\24246\OneDrive\Airport\大兴机场22年航班计划-拖曳计划 - 副本\2022.02航班计划.xlsx"
    _default_tow_path = r"C:\Users\24246\OneDrive\Airport\大兴机场22年航班计划-拖曳计划 - 副本\2月份拖曳计划.xlsx"

    def __init__(self, file_path:str, tow_path:str = None, num_of_data:int = 10000, distribution: bool = False) -> None:
        """
        """
        self.distribution = distribution
        self.num_of_data = num_of_data
        file_path = file_path
        tow_path = tow_path if tow_path != None else RealAirportData._default_tow_path

        """ 以前参数的初始化存在顺序要求 """
        self.near_gates, self.remote_gates = self._load_gates(file_path, num_of_data)

        self.gates_dict = dict[str, Gate]()
        for gate in self.near_gates + self.remote_gates:
            self.gates_dict[gate.name] = gate
        
        # 拖曳模块
        self.tow_module = _TowData(tow_path)

        # __load_flights方法读取到的机位信息为读取的航班而服务
        self.flights, self.artificial_plan = self._load_flights(file_path, num_of_data)

        print(f"|RealAirportData -> __init__()| excel中共读取{len(self.flights)}条航班记录。")

    def get_gates(self, near_or_remote:int, num: Optional[int] = None) -> list[Gate]:
        """Return real designated gates.

        Parameters
        ----------
        near_or_remote:int
            near gates: 0, remote gates: 1.
        num: int | None
            读取的数量，给定None时，读取全部。
        """
        returned_gates = []
        if near_or_remote == 0:
            returned_gates = self.near_gates[:num]
        elif near_or_remote == 1:
            returned_gates = self.remote_gates[:num]
        else:
            raise(ValueError("输入有误"))

        return returned_gates

    def get_flights(self, right:int = -1, left:int = 0, min_date:float = 0, max_date:float = sys.float_info.max, parking_time_threshold:int = 24) -> tuple[list[Flight], list[str]]:
        """
        Parameters
        ----------
        left:int
            航班区间左侧。
        right:int = -1
            航班区间右侧。等于-1时，则表示右侧截止数量与期望读取的excel行数一致。
        min_date:float
            航班区间最小日期。
        max_date:float
            航班区间最大日期。
        parking_time_threshold:int
            机位停留时间阈值，单位为小时。

        Return
        ------
        flights:[Flight]
            航班列表。
        parked_gates_type:[int]
            返回的航班列表的人工分配结果，列表元素的取值为0和1，0表示近机位，1表示远机位。两个列表的顺序一致。
        """

        """ 数据检查 """
        right = right if right != -1 else self.num_of_data
        if type(min_date) != float or type(max_date) != float:
            raise(ValueError("日期数据输入错误"))
        elif min_date > max_date:
            raise(ValueError("时间输入错误"))

        result = [self.flights, self.artificial_plan]
        
        """ 时间筛选 """
        filtered_indices = [i for i, f in enumerate(result[0])
                            if _date_check(min_date, max_date, f) and _check_parking_time(f=f, threshold=parking_time_threshold * 3600)]
        for i in range(len(result)):
            result[i] = [result[i][index] for index in filtered_indices]

        # 航班数量
        if left == -1:
            num = right
        else:
            num = right - left

        f_a = [] # a型机
        f_b = [] # b型机
        f_c = [] # c型机
        f_d = [] # d型机
        f_e = [] # e型机
        # 区分不同机型
        for f, manual_gate in zip(result[0], result[1]):
            f: Flight = f
            if f.size == attribute.attr_size("A"):
                f_a.append((f, manual_gate))
            elif f.size == attribute.attr_size("B"):
                f_b.append((f, manual_gate))
            elif f.size == attribute.attr_size("C"):
                f_c.append((f, manual_gate))
            elif f.size == attribute.attr_size("D"):
                f_d.append((f, manual_gate))
            elif f.size == attribute.attr_size("E"):
                f_e.append((f, manual_gate))
            else: raise(ValueError(f"飞机大小属性值错误，出现该值：{f.size}"))

        # 按比例划取
        # 下方extend的顺序是讲究的，优先添加列表中靠前的航班。
        all_f = []
        all_f.extend(f_c[:int(num * 0.7)])
        all_f.extend(f_e[:int(num * 0.2)])
        all_f.extend(f_d[:int(num * 0.05)])
        all_f.extend(f_b[:int(num * 0.03)])
        all_f.extend(f_a[:int(num * 0.02)])

        """ 数量筛选 """
        flights_distri = list[Flight]()
        manual_gate_distri = list[str]()

        # 装载筛选结果
        for value in all_f:
            flights_distri.append(value[0])
            manual_gate_distri.append(value[1])

        # 如果筛选的数量少了，则从c型机中补充。
        for value in f_c[int(num * 0.8):]:
            if len(flights_distri) < num:
                flights_distri.append(value[0])
                manual_gate_distri.append(value[1])
            else: break

        flights, artificial_plan = result
        flights = flights[left:right]
        artificial_plan = artificial_plan[left:right]
        if self.distribution == True: # 期望按照分布获取航班
            flights = flights_distri
            artificial_plan = manual_gate_distri

        # 按照进港时间排序
        combined = zip(flights, artificial_plan)
        sorted_combined = sorted(combined, key=functools.cmp_to_key(arriving_compare))
        flights, artificial_plan = zip(*sorted_combined)
        return copy.deepcopy(flights), artificial_plan # artificial_plan里面是机位名，str值类型，因此直接传出去即可，而无需deepcopy。

    @_add_distance_info
    def _load_gates(self, file_path, num_of_row):
        """
        Return
        ------
        (near_gates, remote_gates)
        """
        
        df = read_excel("机型大类", "进港机位", "进港机位远近", "出港机位", "出港机位远近", file_path=file_path, num_of_row=num_of_row)

        # 记录每个机位所容纳的最大机型，并将最大机型的大小作为机位的大小。
        gate_size = dict[str, str]()

        # 记录一个机位是近机位还是远机位，如果在数据中出现一个机位又近又远，则报错。
        gates_type = dict[str, str]()

        for data in df:
            flight_type = data[0]
            arrival_gate = data[1]
            arrival_gate_type = data[2]
            departure_gate = data[3]
            departure_gate_type = data[4]

            arr = []
            arr.append((arrival_gate, arrival_gate_type))
            arr.append((departure_gate, departure_gate_type))

            for gate_name, gate_type in arr:
                if gate_name not in gate_size:
                    gate_size[gate_name] = flight_type
                # 新读取到的机型大小超过已有记录的大小
                elif flight_type > gate_size[gate_name]: # 机型的编码为字母A B C D E，因此直接按照ASCII码判断大小即可。
                    gate_size[gate_name] = flight_type

                if gate_name not in gates_type:
                    gates_type[gate_name] = gate_type
                elif gate_type != gates_type[gate_name]:
                    raise(ValueError(f"{type(gate_name), gate_name, gate_type, gates_type[gate_name]} 在文件中，该机位又是近机位，又是远机位。"))

        near_gates = list[Gate]()
        remote_gates = list[Gate]()

        for key, value in gate_size.items():
            gate = Gate(name=key, size=attribute.attr_size(value), airline=attribute.attr_airline(values=True, num=-1), 
                inter_or_domestic=attribute.attr_domestic_and_international(), passenger_or_cargo=attribute.attr_passenger_and_cargo(),
                stand_type=gates_type[key])
            
            if gates_type[key] == "近机位":
                near_gates.append(gate)
            elif gates_type[key] == "远机位":
                remote_gates.append(gate)
            else:
                raise(ValueError(f"机位类型数据错误，读取数据即不是远机位也不是近机位：{gates_type[key]}"))

        num_of_gates = len(near_gates) + len(remote_gates)

        if num_of_gates != len(gate_size):
            raise(ValueError())
        
        near_gates_names = [g.name for g in near_gates]
        remote_gates_names = [g.name for g in remote_gates]
        if num_of_gates != len(set(near_gates_names + remote_gates_names)):
            raise(ValueError("excel中存在相同名字的机位"))

        return near_gates, remote_gates
    
    @_add_passenger_count
    def _load_flights(self, file_path, num_of_row):
        """考虑拖曳的航班信息读取
        
        Return
        ------
        list[Flight]: flight array.
        list[int]: 
        """

        # 拖曳数据模块
        flight_excel = read_excel("ATA", "ATD", "航班号", "机型大类", "进港机位远近", "进港机位", "出港机位远近", "出港机位", "机号", file_path=file_path, num_of_row=num_of_row)
        flights = list[Flight]() # 记录读取到的航班。
        artificial_planed_gates = [] # 记录每架飞机的人工安排机位，以列表的形式，顺序与flights对应。
        for data in flight_excel:
            # 将读取的日期时间转为时间戳
            try:
                ata = TimeMachine.time_to_time_stamp(data[0])
                atd = TimeMachine.time_to_time_stamp(data[1])
            except Exception as _:
                raise(ValueError(f"从Excel中读取的时间存在类型错误, atd: {atd, type(atd)}, ata: {ata, type(ata)}"))

            flight_name = data[2] # 航班号
            flight_size = data[3] # 机型大类
            ar_gate_type = data[4] # 进港机位远近
            arrival_gate = data[5] # 进港机位
            de_gate_type = data[6] # 出港机位远近
            departure_gate = data[7] # 出港机位
            aircraft_Reg_No = data[8] # 航空器注册编号

            arrival_fname, departure_fname = flight_name.split("/")

            # 如果进出港机位与航班表中的机位对不上，则忽略该航班
            if arrival_gate not in self.gates_dict or departure_gate not in self.gates_dict:
                continue
            
            # 剔除进离港机位不同的航班
            # if ar_gate_type != de_gate_type:
            #     continue

            splited_flights = self._split_flights(arrival_fname=arrival_fname, departure_fname=departure_fname, aircraft_Reg_No=aircraft_Reg_No, 
                                                  arrival_time=ata, arrival_gate=arrival_gate, departure_time=atd, departure_gate=departure_gate,
                                                  flight_size=flight_size)
            
            # 如果有且只有一个航班，说明该航班没有拖曳，那么航班号就为进港航班号+出港航班号。
            if len(splited_flights) == 1:
                splited_flights[0][0].flight_No = flight_name ################
            
            """ 3、装载数据 """
            for flight, parked_gate in splited_flights:
                flights.append(flight)
                artificial_planed_gates.append(parked_gate)

        # # 按照停留时间降序
        # flights = sorted(flights, key=lambda x: x.departure_time - x.arrival_time, reverse=True)
        # # 拖曳拆分三段
        # for i in range(20):
        #     pre_f = flight_utils.copy_flight(flights[i])
        #     bak_f = flight_utils.copy_flight(flights[i])
        #     flights[i].departure_time = flights[i].arrival_time + TimeMachine.get_minute(60)
        #     pre_f.arrival_time = flights[i].departure_time
        #     bak_f.arrival_time = pre_f.departure_time - TimeMachine.get_minute(60)
        #     pre_f.departure_time = bak_f.arrival_time
        #     flights.append(pre_f)
        #     flights.append(bak_f)

        return flights, artificial_planed_gates

    def _split_flights(self, **kwargs):
        """根据拖曳数据对航班进行分割，并返回分割的航班与人工安排机位。如果无拖曳则返回航班本身，返回航班的航班名取自flight_name。

        Parameters
        ----------
        aircraft_Reg_No: kwargs["aircraft_Reg_No"]
        departure_fname: kwargs["departure_fname"]
        arrival_fname: kwargs["arrival_fname"]
        arrival_gate: kwargs["arrival_gate"]
        arrival_time: kwargs["arrival_time"]
        departure_gate: kwargs["departure_gate"]
        departure_time: kwargs["departure_time"]
        flight_size: kwargs["flight_size"]

        Return
        ------
        list[list[FLight, str]]: 航班-人工停机位，一维是航班，二维是人工安排机位。
        """

        """ 读取参数 """
        aircraft_Reg_No = kwargs["aircraft_Reg_No"]
        departure_fname = kwargs["departure_fname"]
        arrival_fname = kwargs["arrival_fname"]
        arrival_gate = kwargs["arrival_gate"]
        arrival_time = kwargs["arrival_time"]
        departure_gate = kwargs["departure_gate"]
        departure_time = kwargs["departure_time"]
        flight_size = kwargs["flight_size"]

        """ 加载拖曳数据 """
        tow_data = self.tow_module.get_tow_data(Reg_No=aircraft_Reg_No, flight_name=departure_fname, # 拖曳是以出港航班号为准。
                                               arrival_gate=arrival_gate, arrival_time=TimeMachine.time_stamp_to_time(arrival_time), 
                                               departure_gate=departure_gate, departure_time=TimeMachine.time_stamp_to_time(departure_time))

        """ 读取拖曳时间节点 """
        time_line = []
        for time_point in tow_data:
            # 转换拖曳表中的时间数据的时区
            tow_start_time = time_point[0].tz_localize('Asia/Shanghai').tz_convert('UTC')
            # 标志拖曳目标机位是否为旅客运营机位
            flag = "No" if time_point[3] not in self.gates_dict else "Yes"
            time_line.append([tow_start_time.timestamp(), flag, time_point[4].total_seconds(), time_point[2]])
        # 追加离场信息
        time_line.append([departure_time, "Yes", 0, departure_gate])

        """ 划分航班，根据拖曳时间节点 """
        segmented_flights = []
        start_time = arrival_time
        for end_time in time_line:
            if start_time != None:
                f = Flight.forge_flight(name=departure_fname, aircraft_Reg_No=aircraft_Reg_No, 
                                        at=start_time, dt=end_time[0], size=flight_size)
                segmented_flights.append([f, str(end_time[3])]) # 航班-人工停机位
                start_time = None

            if end_time[1] == 'Yes':
                start_time = end_time[0] + end_time[2]

        # 分割后的第一个航班属于进港航班，因此这里换上进港航班号。
        segmented_flights[0][0].flight_No = arrival_fname ###############################

        return segmented_flights


class _TowData:
    """读取拖曳数据
    """

    # 拖曳文件中所需读取的列
    AIRCRAFT_REG_NO = "机号(A)"
    FLIGHT_NAME = "航班号(D)"
    ORG_GATE = "机位(Org)"
    TAR_GATE = "机位(Des)"
    START_TIME = "实际开始(TOW)"
    END_TIME = "实际结束(TOW)"
    TIME_CONSUMED = "实际耗时"

    def __init__(self, file_path) -> None:
        # 表格读取 DataFrame
        tow_df = pd.read_excel(file_path, engine='openpyxl')
        tow_df = tow_df.fillna("BAD DATA")
        # 表格数据处理
        tow_df[self.START_TIME] = pd.to_datetime(tow_df[self.START_TIME])
        tow_df[self.END_TIME] = pd.to_datetime(tow_df[self.END_TIME])
        tow_df[self.TIME_CONSUMED] = tow_df[self.TIME_CONSUMED].astype(int)
        tow_df[self.ORG_GATE] = tow_df[self.ORG_GATE].astype(str)
        tow_df[self.TAR_GATE] = tow_df[self.TAR_GATE].astype(str)
        self.tow_df = tow_df

    def get_tow_data(self, **kwargs):
        """检查飞机的拖曳情况，并按照时间顺序返回其所有拖曳时间点。

        Parameters
        ----------
        Reg_No: str, 航班的航空器注册编号。
        flight_name: str, 航班号。
        arrival_time: str, 航班的进港时间，格式为2002-02-01 00:00:00。
        departure_time: str, 航班的出港时间，格式为2002-02-01 00:00:00。
        arrival_gate: str, 航班的进港机位。
        departure_gate: str, 航班的出港机位。

        Note
        -----
        所有这些参数都必须提供。
        """

        # 参数检查
        if type(kwargs["arrival_time"]) != str or type(kwargs["departure_time"]) != str:
            raise(ValueError(f"输入时间有误。"))
        
        time_points = []

        # 筛选数据
        filtered_df = self._filter_df(**kwargs)
        
        # 检查数据
        check_flag = _TowData._check_df(df=filtered_df, **kwargs)

        # 读取数据
        if check_flag == True:
            start_times = filtered_df[_TowData.START_TIME]
            end_times = filtered_df[_TowData.END_TIME]
            org_gate = filtered_df[_TowData.ORG_GATE]
            tar_gate = filtered_df[_TowData.TAR_GATE]
            time_consumed = filtered_df[_TowData.TIME_CONSUMED]
            for i in range(len(filtered_df)):
                time_points.append([start_times.iloc[i], end_times.iloc[i], org_gate.iloc[i], 
                                    tar_gate.iloc[i], pd.Timedelta(minutes=int(time_consumed.iloc[i]))])
        return time_points
    
    def _filter_df(self, **kwargs):
        """DataFrame数据筛选，返回的df按照计划结束（TOW）时间升序。
        """

        # 参数读取
        df = self.tow_df
        aircraft_Reg_No = kwargs["Reg_No"]
        flight_name = kwargs["flight_name"]
        arrival_time = kwargs["arrival_time"]
        departure_time = kwargs["departure_time"]

        # 日期筛选
        filtered_df = df[(arrival_time <= df[_TowData.START_TIME]) & (df[_TowData.END_TIME] <= departure_time)]
        # 航空器注册号与航班号筛选
        filtered_df = filtered_df[(filtered_df[_TowData.FLIGHT_NAME] == flight_name) & (filtered_df[_TowData.AIRCRAFT_REG_NO] == aircraft_Reg_No)]
        # 按照拖曳开始时间进行排序，升序
        filtered_df = filtered_df.sort_values(by='实际开始(TOW)')

        return filtered_df
    
    @staticmethod
    def _check_df(df:pd.DataFrame, **kwargs):
        """检查拖曳数据是否与航班数据相对应。
        """

        """ 信息收集 """
        # 航班的进港机位和出港机位
        arrival_gate = "arrival_gate"
        departure_gate = "departure_gate"
        # "实际开始(TOW)" - "实际结束(TOW)"，得到的时间单位为分钟。
        time_consumed = (df[_TowData.END_TIME] - df[_TowData.START_TIME]).dt.total_seconds() / 60
        # 检查拖曳表中，实际开始 实际结束 实际耗时，这三列的值能不能对上。
        consume_compareison = time_consumed != df[_TowData.TIME_CONSUMED]
        
        # 数据检查标志，合格为True，不合格False。
        returned_flag = True
        no_pass = lambda: returned_flag * False

        """ 进行检查。共四条检查规则 """
        if len(df) < 1: returned_flag = no_pass() # 没有拖曳数据
        elif kwargs[arrival_gate] != df[_TowData.ORG_GATE].iloc[0]: returned_flag = no_pass() # 航班的进港机位与第一次拖曳的始发机位不同，则数据不合理。
        elif kwargs[departure_gate] != df[_TowData.TAR_GATE].iloc[-1]: returned_flag = no_pass() # 航班的出港机位与最后一次拖曳的目标机位不同，则数据不合理。
        elif consume_compareison.any(): returned_flag = no_pass() # 拖曳表中，拖曳耗时数据错误，对不上，给了两倍的冗余都对不上。
        else: pass

        return returned_flag

