import time
import functools


def measure_execution_time(func):
    """
    计算函数的执行时间。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        
        return result, round(end_time - start_time, 2)
        # return result, end_time - start_time
    return wrapper

