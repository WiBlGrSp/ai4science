### 重构特征扩展模块

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

# ===================== 全局参数配置 =====================
num_epochs, batch_size, lr, num_expand, use_net = 10, 16, 1e-2, 5, False

# 设备配置：优先使用GPU，无GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 项目运行根目录
path = os.getcwd()
# path = "yourfilepath"

# ===================== 主程序执行流程 =====================
print("------------ 程序开始运行 ------------")

# 读取输入数据与目标变量
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

# 对原始数据进行特征空间扩展
data_expanded = expand(data,num_expand)

# 确定性独立筛选：提取高相关性特征
data_sis = sis(data_expanded, focus.to_numpy(), 10)

# 启动计时器，统计拟合耗时
ti = Timer()

# 模型系数拟合与训练
r2, coef, loss = fit(data_sis, focus.to_numpy(), device,
                     use_net=use_net,
                     num_epochs=num_epochs,batch_size=batch_size,lr=lr)

# 停止计时并输出运行时间
ti.stop()
print(ti.cumsum())

# 对特征拟合结果进行排序整理
results= sort_result(data_sis.to_numpy(), 
                      data_sis.columns.to_numpy(),
                      r2, coef, loss, 10)

# 将运行参数与时间写入日志文件
with open(os.path.join(path,"log"),'w', encoding='utf-8') as file:
    file.write(str(ti.cumsum())+'\n')
    file.write(f"{num_epochs=},{batch_size=},{lr=}"+'\n')
    file.write("符号空间大小="+str(len(data_expanded.columns)))

# 展示排序后的特征结果
display_result(results, 10, path)

# 绘制最优特征拟合图像
plot_result(results[0], focus, path)