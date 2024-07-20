"""
   画图 
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline

from . import flight_utils
from ..model.gate_graph import Flight
from .time_machine import TimeMachine
from .file import create_folder

# matplotlib.rc('font', family='FangSong', weight='bold', size=14)
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


def draw_gantt(gates_flights_dict:dict[str, list[Flight]], min_date:float = None, max_date:float = None,
               additional_infor:list[str]=None, title='', display_name:bool=True, save_path:str=None):
    """
        画甘特图

    Parameters
    ----------
    save_path: str
        例如：C:/Users/surface/OneDrive/Airport/Code/
    """
    plt.close('all')

    if type(display_name) != bool:
        raise(ValueError(f"参数输入错误{display_name}"))

    flights = list[Flight]()
    for _, value in gates_flights_dict.items():
        flights.extend(value)

    # 创建绘图对象
    fig, ax = plt.subplots(figsize=(12, 8), dpi=2560 / 10, constrained_layout=True)
    
    if additional_infor != None:
        infor_arr = []
        if type(additional_infor) != list:
            infor_arr.append(additional_infor)
        else:
            infor_arr.extend(additional_infor)

        # 初始位置设置
        x = 0
        p = 0
        horizontalalignment = "left"
        for i, v in enumerate(infor_arr):
            if i == 4:
                x = 1
                p = 0
                horizontalalignment = "right" 
            y_bias = p * 0.05
            p += 1
            ax.text(x, 1.15 + y_bias, f"{v}", 
                verticalalignment='top', horizontalalignment=horizontalalignment,  
                transform=ax.transAxes,  
                fontsize=18)  # fontsize定义了字体的大小 
    
    # 设置Y轴刻度为机位号
    ax.set_yticks(range(len(gates_flights_dict))) # “打桩”
    ax.set_yticklabels(gates_flights_dict.keys(), fontsize=6) # “给桩打标签”

    # 根据数据中的最小日期和最大日期设置X轴范围
    min_date = flight_utils.sort_flights(flights)[0].arrival_time - 600 # 减600是为了让图的左边缘留出空隙.
    max_date = flight_utils.sort_flights(flights, flag=1)[-1].departure_time + 600 # 加600让图的右边缘留出空隙.
    min_date = TimeMachine.time_stamp_2_datatime(min_date)
    max_date = TimeMachine.time_stamp_2_datatime(max_date)
    ax.set_xlim(min_date, max_date)
    
    flight_num = 0
    for i, (_, flights) in enumerate(gates_flights_dict.items()):
        for f in flights:
            flight_num += 1
            f: Flight = f

            # 航班在图上的长度
            duration_hour = TimeMachine.seconds_2_hour(f.departure_time - f.arrival_time)
            duration_timedelta = TimeMachine.hours_2_timedelta(duration_hour)

            # arrival_time: 航班在图上的起始位置
            arrival_time = TimeMachine.time_stamp_2_datatime(f.arrival_time)
            ax.barh(i, duration_timedelta, left=arrival_time, height=0.3, align='center', alpha=0.8)

            # 添加航班号标注
            if display_name == True:
                flight_label = f.flight_No
                ax.annotate(flight_label, xy=(arrival_time, i ), xytext=(10, 0), textcoords='offset points',
                            ha='left', va='center', fontsize=6)

    print(f"|draw_graph -> draw_gantt()| 共绘制{flight_num}架航班")
    # 设置图形标题和轴标签
    #title = "机位占用图" + title
    t = f"{title}"
    # 设置日期时间格式化器
    ax.set_title(t)
    # ax.text(0.5, -0.13, 'OG: 普通机位; FG: 父子机位; SFG: 顺序父子机位; DSS: 减容机位的小机位; DSB: 减容机位的大机位; L: 左子机位; R: 右子机位', 
    #         ha='center', va='top', transform=ax.transAxes)
    # ax.set_xlabel('时间')

    # 设置日期时间格式化器
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 调整x轴字体大小
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=18)

    # 自动调整日期显示格式
    # fig.autofmt_xdate()

    if save_path == None:
        plt.show()

    file_name = "甘特图 " + title
    if display_name: 
        file_name += " 带航班名"
    else:
        file_name += " 不带航班名"

    _save_fig(fig=fig, save_path=save_path, file_name=file_name)


def horizontal_bar_chart(y_labels: list[str], y_datas: list[float], title, x_labels, file_name, save_path):
    plt.close('all')
    y_datas = np.array(y_datas)
    sorted_indices = np.argsort(y_datas)
    y_labels = np.array(y_labels)[sorted_indices]
    y_datas = y_datas[sorted_indices]

    fig, ax = plt.subplots()

    ax.barh(y_labels, y_datas, align='center') 
    ax.set_xlabel(x_labels)
    ax.set_title(title)

    _save_fig(fig=fig, save_path=save_path, file_name=file_name)
    

def plot_scatter(x: list, y: list, xlable: str, ylable: str, title: str=None, file_name=None, save_path=None):
    """
    绘制散点图

    Parameters
    ----------
    x : list[int | float]
    y : list[int | float]
    xlable: str
    ylable: str
    title : str | None
    file_name : str | None
    save_path : str | None
    """
    plt.close('all')
    if len(x) != len(y):
        raise(ValueError(f"data error. len of x is {len(x)}, and of y is {len(y)}"))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=2560 / 10)
    for xi, yi in zip(x, y):
        ax.scatter(xi, yi)
    title = title if title != None else "Scatter Plot"
    # Add labels and title
    ax.set_xlabel(xlable)
    ax.set_ylabel(ylable)
    ax.set_title(title)

    # Show plot
    if save_path == None:
        plt.show()

    _save_fig(fig=fig, save_path=save_path, file_name=file_name)
    

def plot_line_chart(x_data: list=None, y_data: list=None, labels: list[str]=None, save_path=None, 
                    x_label: str='x', y_label: str='y', title: str='Line Chart', x_ticks: list=None, y_ticks: list=None):
    plt.close('all')
    plt.rcParams.update({'font.size': 14})
    # 设置宋体字体
    plt.rcParams['font.family'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False # 用于正常显示负号
    # 长度不一，数据不对称，报错。
    data_len = []
    data_len.append(len(x_data))
    if type(y_data[0]) == list:
        for y in y_data:
            data_len.append(len(y))
    else:
        data_len.append(len(y_data))
        y_data = [y_data]
    if len(set(data_len)) != 1:
        raise(ValueError(f"data error. len of x_data is {len(x_data)}, len of y_data is {len(y_data)}, and len of labels is {len(labels)}"))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1440 / 10, layout='constrained')

    symbol = ['b--', 'g--', 'g-*', 'm**', 'c-+']

    for i, y in enumerate(y_data):
        ax.plot(x_data, y, symbol[i], alpha=0.8, linewidth=2, label=labels[i])

    ax.legend(frameon=False)  #显示上面的label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)#accuracy
    ax.set_title(title)
    ax.set_xlim([min(x_data), max(x_data)])
    if x_ticks != None:
        _ = ax.set_xticks(x_ticks)
    if y_ticks != None:
        _ = ax.set_yticks(y_ticks)
    # ax.set_yticks([68, 70, 72, 74, 76, 78])
    # yticklabels = [item.get_text() for item in ax.get_yticklabels()]
    # yticklabels[0] = '' # 去除第一个刻度尺
    # ax.set_yticklabels(yticklabels)
    ax.tick_params(direction='in')
    ax.set_box_aspect(0.8)

    # plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)

    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.savefig(r'output.svg', format='svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_smooth_line_chart(x_data: list=None, y_data: list=None, labels: list[str]=None, save_path=None, 
                    x_label: str='x', y_label: str='y', title: str='Line Chart', x_ticks: list=None, y_ticks: list=None):
    plt.close('all')
    plt.rcParams.update({'font.size': 14})
    # 设置宋体字体
    plt.rcParams['font.family'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False # 用于正常显示负号

    # 长度不一，数据不对称，报错。
    data_len = []
    data_len.append(len(x_data))
    if type(y_data[0]) == list:
        for y in y_data:
            data_len.append(len(y))
    else:
        data_len.append(len(y_data))
        y_data = [y_data]
    if len(set(data_len)) != 1:
        raise(ValueError(f"data error. len of x_data is {len(x_data)}, len of y_data is {len(y_data)}, and len of labels is {len(labels)}"))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1440 / 10, layout='constrained')

    symbol = ['b-', 'r--', 'g-*', 'm**', 'c-+']

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    for i, y in enumerate(y_data):
        # 创建插值模型
        spline = make_interp_spline(x_data, y)
        # 生成更密集的 x 轴值数组，但保持原样的 x 轴范围
        x_smooth = np.linspace(x_data.min(), x_data.max(), len(x_data) * 60)
        y_smooth = spline(x_smooth)
        # 如果要使用插值平滑模型，则将下面的 y 替换为 y_smooth，x_data 替换为 x_smooth。
        if labels == None:
            ax.plot(x_data, y, symbol[i], alpha=0.8, linewidth=2)
        else: 
            ax.plot(x_data, y, symbol[i], alpha=0.8, linewidth=2, label=labels[i])

    ax.legend(frameon=False)  #显示上面的label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)#accuracy
    ax.set_title(title)
    ax.set_xlim([min(x_data), max(x_data)])
    if x_ticks != None:
        _ = ax.set_xticks(x_ticks)
    if y_ticks != None:
        _ = ax.set_yticks(y_ticks)
    ax.tick_params(direction='in')
    ax.set_box_aspect(0.8)

    plt.savefig(r'smooth_line_output.svg', format='svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def grouped_bar_chart(species: tuple, penguin_means: dict[str, tuple[float]], y_label: str=None, title: str=None, 
                      max_y_value: float=None, save_path=None, file_name=None):
    """
    species = ("Adelie", "Chinstrap", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }
    """

    plt.close('all')
    plt.rcParams.update({'font.size': 15})

    species = species
    penguin_means = penguin_means

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(dpi=1440 / 10, layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=13)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    y_label = y_label if y_label != None else '未设置y轴标签'
    title = title if title != None else '未设置标题'
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper right', ncols=3)
    if max_y_value != None:
        ax.set_ylim(0, max_y_value)

    plt.savefig(r'grouped_bar_chart.svg', format='svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def _save_fig(fig, save_path, file_name):
    if save_path != None:
        # 创建文件夹然后保存文件
        save_path = save_path.replace('/', '\\')
        create_folder(floder_path=save_path)
        fig_path = f"{save_path}{file_name}.svg"
        print(f"|draw_graph -> _sava_fig()| {file_name} 保存路径：{save_path}")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig=fig)

