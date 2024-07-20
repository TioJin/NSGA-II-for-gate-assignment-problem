import time
import random
from datetime import datetime, timedelta

"""
    日期格式与时间戳格式互转
"""

class TimeMachine:

    @staticmethod
    def format_time():
        pass

    @staticmethod
    def get_time_stamp(year=time.strftime("%Y", time.localtime()), 
                        month=time.strftime("%m", time.localtime()), 
                        day=time.strftime("%d", time.localtime()), 
                        hour=None,
                        minute=None,
                        second=0) -> float:
        """
            给定日期，返回时间戳。时间格式为 2023-11-07 20:24:00。
            日期的各个部分可自定义。年月日如果不自定义则采取系统时间，小时和分钟采取随机时间。
        """
        hour = random.randint(0, 23) if hour is None else hour
        minute = random.randint(0, 59) if minute is None else minute
        t = f"{year}-{month}-{day} {hour}:{minute}:{second}"
        time_stamp = TimeMachine.time_to_time_stamp(t)
        return time_stamp

    @staticmethod
    def get_time_slot(year=time.strftime("%Y", time.localtime()), 
                        month=time.strftime("%m", time.localtime()), 
                        day=time.strftime("%d", time.localtime()), 
                        hour=None,
                        minute=None,
                        second=0,
                        min_interval=5,
                        max_interval=30) -> tuple:
        """
            返回一段随机时间戳。min_interval为最短间隔时间，max_interval为最长间隔时间，随机时间在这个范围内产生。
        """
        start_time = TimeMachine.get_time_stamp(year, month, day, hour, minute, second)
        interval = random.randint(min_interval, max_interval)
        end_time = start_time + interval * 60
        return (start_time, end_time)
    
    @staticmethod
    def time_to_time_stamp(t: str):
        """"%Y-%m-%d %H:%M:%S"日期格式转时间戳
        """
        return time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def time_stamp_to_time(time_stamp: float):
        """时间戳转日期
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_stamp))

    @staticmethod
    def seconds_2_hour(seconds: int) -> int:
        return seconds / 3600
    
    @staticmethod
    def seconds_2_minute(seconds: int) -> int:
        return seconds / 60
    
    @staticmethod
    def gett(time_stamp) -> int:
        """时间戳转小时
        """
        dt_object = datetime.datetime.fromtimestamp(time_stamp)  
        # 提取小时、分钟和秒  
        hour = dt_object.hour  
        minute = dt_object.minute  
        second = dt_object.second 

        time = hour + minute / 60 + second / 3600
        return time

    @staticmethod
    def get_minute(minute: float) -> float:
        """给定分钟，返回时间戳格式。
        """
        if minute < 0:
            raise ValueError("时间输入错误，分钟取值不应小于0。")
        return minute * 60

    @staticmethod
    def time_stamp_2_datatime(time_stamp):
        str_data = TimeMachine.time_stamp_to_time(time_stamp)
        return datetime.strptime(str_data, '%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def hours_2_timedelta(hours):
        return timedelta(hours=hours)

    @staticmethod
    def check_time_overlap(start1: float, end1: float, start2: float, end2: float):
        """
        判断两个时间段是否存在重叠。

        :param start1: 第一个时间段的开始时间
        :param end1: 第一个时间段的结束时间
        :param start2: 第二个时间段的开始时间
        :param end2: 第二个时间段的结束时间
        :return: 如果时间段重叠，返回True以及重叠时间段；否则返回False以及中间的间隔时间段。
        """
        """ 参数格式检查 """
        if start1 > end1 or start2 > end2:
            raise(ValueError(f"时间输入错误。{start1}，{end1} --- {start2}，{end2}"))
        
        """ 时间重叠判断 """
        latest_start = max(start1, start2)
        earliest_end = min(end1, end2)
        overlap = latest_start < earliest_end
        return overlap, (latest_start, earliest_end) if overlap else (earliest_end, latest_start)

if __name__ == "__main__":
    print(TimeMachine.time_to_time_stamp("2022-02-01 00:00:00"))
