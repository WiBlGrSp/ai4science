# 导入依赖库
import os
import pandas as pd
from expansion import expand
from expansion import expand_next
from sis_ana import sis  # 确定性独立筛选
from coefficients_fitting import fit  # 系数拟合
from result_sorting import sort_result_and_best  # 结果排序
from result_displaying import display_result  # 结果输出
import torch
from feature_information import FeatureInformation
import itertools
import numpy as np


# 程序启动标识
print("===============程序开始================p")

# -------------------------- 1. 数据读取与设备配置 --------------------------
# 配置计算设备：GPU优先，无GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取输入特征数据与目标输出数据
data = pd.read_csv('data.csv')
focus = pd.read_csv('focus.csv')

# 获取当前代码运行的工作目录
path = os.getcwd()

# -------------------------- 2. 迭代终止函数定义 --------------------------
def stop_or_not(iter_num, current_score, best_score, patience=5):
    """
    迭代终止判断函数
    :param iter_num: 当前迭代次数
    :param current_score: 当前最优R²分数
    :param best_score: 历史最优R²分数
    :param patience: 无提升最大迭代次数
    :return: True-终止迭代;False-继续迭代
    """
    # 策略1：达到最大迭代次数终止
    max_iter = 20
    if iter_num >= max_iter:
        return True
    # 策略2：超过patience次迭代无分数提升则终止
    if current_score <= best_score and iter_num > patience:
        return True
    return False


# -------------------------- 3. 循环迭代核心逻辑 --------------------------
# 初始化迭代参数
iter_num = 1                # 当前迭代次数
continue_flag = True        # 循环控制标志
all_results = []           # 存储所有迭代结果
best_r2 = 0                # 历史最优r2分数
results_cur = None         # 当前轮次临时结果

# 清空日志文件，准备写入新的运行记录
with open(os.path.join(path,"log"),'w', encoding='utf-8') as file:
    pass

# 迭代特征工程主循环
while continue_flag:
    # 1. 特征空间扩展：首轮使用全量扩展，后续使用迭代式扩展
    if iter_num == 1:
        data_expanded = expand(data)
    else:
        data_expanded = expand_next(data, best_feature)

    # 2. 确定性独立筛选（SIS），对特征空间进行降维
    data_sis = sis(data_expanded, focus.to_numpy(), 10)

    # 3. 模型系数拟合（支持神经网络训练）
    r2, coef, loss = fit(data_sis, focus.to_numpy(), device=device, use_net=True)

    # 4. 结果排序，提取当前轮次最优特征
    results, best_feature = sort_result_and_best(data_sis.to_numpy(), 
                            data_sis.columns.to_numpy(),
                            r2, coef, loss, 10)

    # 更新当前轮次最优r2与全局最优r2
    cur_best_r2 = max(r2)
    best_r2 = max(best_r2, cur_best_r2)
    
    # 打印并记录本轮结果
    with open(os.path.join(path, 'log'), 'a', encoding='utf-8') as file:
        file.write(f"第{iter_num}轮结果:\n")
    display_result(results, 1e9, path)
    
    # 判断是否满足终止条件
    if stop_or_not(iter_num, cur_best_r2, best_r2):
        continue_flag = False
    else:
        iter_num += 1