"""
    此模块负责静态约束的检查。

    该模块高度依赖属性值模块，属性值检查嘛，不知道属性值怎么检查。
"""

from abc import ABC, abstractmethod

from ..model.gate_graph import Gate, Flight

# 检查器
class ConstrainCheck(ABC):
    @abstractmethod
    def check(gate: Gate, flight: Flight) -> bool:
        pass


# 机位-机型机位容量约束
class _SizeCheck(ConstrainCheck):

    def check(self, gate: Gate, flight: Flight) -> bool:
        if flight.size <= gate.size:
            return True
        return False


# 机位-航司（可突破，但尽量不要出现）机位航司约束
class _AirlineCheck(ConstrainCheck):

    def check(self, gate: Gate, flight: Flight) -> bool:
        if isinstance(gate.airline, list):
            # 机位的列表为空代表没有航司偏好。count不等于0代表机位的航司偏好中包含该航班的航司。
            if len(gate.airline) == 0 or gate.airline.count(flight.airline) != 0:
                return True
        else:
            raise
        return False


# 机位-属性（国际国内）（可突破，但尽量不要出现）
class _InterORdomesticCheck(ConstrainCheck):

    def check(self, gate: Gate, flight: Flight) -> bool:
        if gate.inter_or_domestic >= flight.inter_or_domestic:
            return True
        return False
    

# 机位-任务（客货运）（可突破，但尽量不要出现）
class _PassengerORcargoCheck(ConstrainCheck):

    def check(self, gate: Gate, flight: Flight) -> bool:
        if gate.passenger_or_cargo >= flight.passenger_or_cargo:
            return True
        return False


# 模块外调用此方法即可进行静态约束检查
def check(gate: Gate, flight: Flight) -> bool:
    """
    静态约束检查，通过则返回True，否则返回False
    """

    for checker in constarin_checker:
        # 有一条属性不通过，则返回False
        if not checker.check(gate, flight):
            return False
    # 全部通过，返回True
    return True


constarin_checker = [
    _AirlineCheck(), 
    _SizeCheck(), 
    _InterORdomesticCheck(), 
    _PassengerORcargoCheck()
    # 上面新增一个检查器，则在这里增加其实例
]

