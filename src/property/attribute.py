"""
    整合属性。
"""


import random
import numpy as np
from typing import Literal


# 属性有较多值时，使用此方法为属性构建字典并赋予常数值
def __attribute_maker(attr_names: list[str]) -> dict:
    attr_dict = {}
    for i in range(len(attr_names)):
        attr_dict[attr_names[i]] = i
    return attr_dict


# 容量大小/机型大小
_size = np.array((["A", "B", "C", "D", "E", "F"]), dtype=str)
_sizes_dict = __attribute_maker(_size)

# 东航、南航、国航、厦航、海南航空、瑞丽航空
_airlines = np.array((["china eastern", "china southern",
                    "air china", "xiamen", "hainan", "ruili"]), dtype=str)
_airlines_dict = __attribute_maker(_airlines)

# 国内
_domestic = 0
# 国际
_international = 1
# 机位专用属性。表示国内和国际都可以停在此机位。
_inter_and_domestic = 2

# 乘客
_passenger = 0
# 货物
_cargo = 1
# 机位专用属性。表示乘客和货物都可以停在此机位。
_passenger_and_cargo = 2


def attr_size(s: str) -> int:
    """
        根据飞机大小名来返回大小值。"A", "B", "C", "D", "E", "F"。
    """
    if type(s) != str:
        raise ValueError("参数类型错误")
    return _sizes_dict[s]


def size_to_str(s: int) -> str:
    """根据size的int值来返回其对应的str值
    """
    return _size[s]


def attr_random_size() -> int:
    """
        随机获取一个机型大小
    """
    return random.choice(list(_sizes_dict.values()))


def get_maxsize() -> int:
    return list(_sizes_dict.values())[len(_sizes_dict) - 1]


def attr_airline(name=None, values: bool = False, num: int = 0):
    """
        给定name返回对应值；设置values=True则返回航司数组。

        如果name为str类型，则代表航司名，方法会返回对应的int类型航司值。

        如果name为int类型，则代表航司值，方法会返回对应的航司str类型名。

        如果name等于None，则随机返回一个int类型航司值。

        如果values为True，则返回一个包含int类型航司值的数组，其长度为num，num=0则长度随机，可赋予其int值来指定长度，如果num=-1，则返回所有航司。
    """
    airline = None
    if values == False:
        if isinstance(name, str):
            airline = _airlines_dict.get(name)
        elif isinstance(name, int):
            airline = _airlines[name]
        elif airline is None:
            airline = random.choice(list(_airlines_dict.values()))

    else:
        if num == -1:
            num = len(_airlines_dict)
        elif num == 0:
            num = random.randint(1, len(_airlines_dict) - 1)
        airline = random.sample(list(_airlines_dict.values()), num)

    return airline


def airline_to_str(airline:int) -> str:
    """将航司的int值转为航司名称
    """
    return _airlines[airline]


def attr_domestic() -> int:
    return _domestic


def attr_international() -> int:
    return _international


def attr_domestic_and_international() -> int:
    """
        返回国内和国际都可以停的属性值。
    """
    return _inter_and_domestic


def attr_random_domes_inter(single=True) -> int:
    """
        随机返回国内或国际属性。

        如果single=False，则随机返回国内、国际或国内国际属性。
    """
    values = [attr_domestic(), attr_international()]
    if single == False:
        values.append(attr_domestic_and_international())
    return random.choice(values)


def attr_passenger() -> int:
    return _passenger


def attr_cargo() -> int:
    return _cargo


def attr_passenger_and_cargo() -> int:
    """
        返回“客货”都可以停的属性值。
    """
    return _passenger_and_cargo


def attr_random_passenger_cargo(single=True) -> int:
    """
        随机返回客运或货运属性。

        如果single=False，则随机返回客运、货运或客运货运属性。
    """
    values = [attr_passenger(), attr_cargo()]
    if single == False:
        values.append(attr_passenger_and_cargo())
    return random.choice(values)


def get_child_size(size) -> str:
    """
        根据父机位大小返回子机位大小
    """
    """
        目前子机位的大小为父机位降一级
    """
    if isinstance(size, str):
        size = attr_size(size)
    if size <= 0:
        # 父机位为最小机位时，无法分为两个子机位。
        raise ValueError("父机位大小不合适")
    return size - 1 


def is_passenger_and_cargo(value):
    """
        判断机位是否为货运客运都可以停
    """
    if value == _passenger_and_cargo:
        return True
    return False


def is_domestic_and_international(value):
    """
        判断机位是否为国内国际都可以停
    """
    if value == _inter_and_domestic:
        return True
    return False


# print(attr_airline(name=0))
# print(attr_airline(values=True, num=-1))
# print(attr_random_domes_inter(single=False))
# print(attr_random_passenger_cargo(single=False))
# print(attr_random_size())
# print(get_maxsize())
# print(get_child_size(1))

