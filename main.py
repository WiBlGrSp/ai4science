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

# 参数
num_epochs=10
batch_size=16
lr=1e-2
num_expand=5 #扩展元数
use_net = False #使用神经网络

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代码运行目录
path = os.getcwd()
# path = "yourfilepath"

print("------------ 程序开始运行 ------------")

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 初始输入的扩充
data_expanded = expand(data,num_expand)
# check_bad_columns(data_expanded)

# 确定性独立筛选
data_sis = sis(data_expanded, focus.to_numpy(), 10)
# data_sis= data_expanded.dropna(axis=1)
# 系数拟合
ti = Timer()

r2, coef, loss = fit(data_sis, focus.to_numpy(), device,
                     use_net=use_net,
                     num_epochs=num_epochs,batch_size=batch_size,lr=lr)

ti.stop()
print(ti.cumsum())

# 结果整理
results= sort_result(data_sis.to_numpy(), 
                      data_sis.columns.to_numpy(),
                      r2, coef, loss, 10)
# 输出日志
with open(os.path.join(path,"log"),'w', encoding='utf-8') as file:
    file.write(str(ti.cumsum())+'\n')
    file.write(f"{num_epochs=},{batch_size=},{lr=}"+'\n')
    file.write("符号空间大小="+str(len(data_expanded.columns)))
display_result(results, 10, path)

# 绘制图像
plot_result(results[0], focus, path)
