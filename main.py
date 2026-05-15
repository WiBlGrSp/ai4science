from expansion import expand
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit 
from result_plotting import plot_result
import pandas as pd
import torch
import os
from timer import Timer

from tools import check_bad_columns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代码运行目录
path = os.getcwd()
# path = "yourfilepath"

print("------------ 程序开始运行 ------------")

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 初始输入的扩充
data_expanded = expand(data)
check_bad_columns(data_expanded)

# 确定性独立筛选
# data_sis = sis(data_expanded, focus.to_numpy(), 10)
# data_sis= data_expanded.dropna(axis=1)
data_sis = data_expanded
# 系数拟合
ti = Timer()

r2, coef, loss = fit(data_sis, focus.to_numpy(), device)

ti.stop()
print(ti.cumsum())

# 结果整理
results = sort_result(data_sis.to_numpy(), 
                      data_sis.columns.to_numpy(),
                      r2, coef, loss, 1050)

# 输出日志
display_result(results, 1050, path)

# 绘制图像
plot_result(results[0], focus, path)
