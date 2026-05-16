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


print("===============程序开始================p")
# -------------------------- 1. 数据读取 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取输入特征数据与目标输出数据
data = pd.read_csv('data.csv')
focus = pd.read_csv('focus.csv')
# 代码运行目录
path = os.getcwd()

# --------------------------  迭代终止函数定义 --------------------------
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
    max_iter = 5
    if iter_num >= max_iter:
        return True
    # 策略2：连续patience次无分数提升终止
    if current_score <= best_score and iter_num > patience:
        return True
    return False


# --------------------------  循环迭代核心逻辑 --------------------------
# 初始化参数
iter_num = 1  # 迭代次数
continue_flag = True  # 循环控制标志
all_results = []  # 存储所有迭代结果
best_r2 = 0 #历史最优r2
results_cur = None  # 当前轮次临时结果

with open(os.path.join(path,"log"),'w', encoding='utf-8') as file:
    pass

while continue_flag:
    #1.特征空间扩展：首轮用初始扩展，后续用迭代式针对性扩展
    if iter_num == 1:
        data_expanded = expand(data)
    else:
        data_expanded = expand_next(data,best_feature)

    #2.确定性独立筛选（SIS）降维
    data_sis = sis(data_expanded, focus.to_numpy(),10)

    #3.系数拟合（解析法/神经网络法）
    r2, coef, loss = fit(data_sis, focus.to_numpy(),device=device,use_net=True)

    #4.结果排序，提取当前轮次最优特征
    results, best_feature = sort_result_and_best(data_sis.to_numpy(), 
                            data_sis.columns.to_numpy(),
                            r2, coef, loss, 10)

    # 更新最优分数
    cur_best_r2 = max(r2)
    best_r2 = max(best_r2,cur_best_r2)
    
    #打印本轮结果
    with open(os.path.join(path, 'log'), 'a', encoding='utf-8') as file:
        file.write(f"第{iter_num}轮结果:\n")
    display_result(results,1e9,path)
    
    # 判断是否终止迭代
    if stop_or_not(iter_num, cur_best_r2, best_r2):
        continue_flag = False
    else:
        iter_num += 1

# -------------------------- 5. 最终结果输出 --------------------------

