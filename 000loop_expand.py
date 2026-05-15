# 导入依赖库
import os
import pandas as pd
from expansion import expand  # 初始特征扩展
from sis_ana import sis  # 确定性独立筛选
from coefficients_fitting import fit  # 系数拟合
from result_sorting import sort_result_and_best  # 结果排序
from result_displaying import display_result  # 结果输出
import torch
from feature_information import FeatureInformation
from expansion import combine
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
    # 策略2：连续patience次无分数提升终止
    if current_score <= best_score and iter_num > patience:
        return True
    return False

# -------------------------- 3. 结果存储函数定义 --------------------------
def store_result(results_temp:list[FeatureInformation], all_results:list[FeatureInformation]):
    """
    存储每轮迭代的最优结果
    :param results_temp: 当前轮次结果
    :param all_results: 历史所有结果
    :return: 更新后的总结果
    """
    all_results.extend(results_temp)
    return all_results


def get_contained_chars(col_name: str, char_list: list[str]) -> set:
    """
    输入一个列名，返回它包含的所有符号（集合）
    """
    col_name = str(col_name)
    return {char for char in char_list if char in col_name}


#迭代扩展符号空间,将上一轮最优特征加入到原扩展空间，获得新扩展空间
#参数:上次经过扩展的符号空间，上次topn的最优特征结果，上次经过sis的符号空间
def expand_next(data_expended:pd.DataFrame,in_feature:pd.DataFrame)->pd.DataFrame:
    ## 初始化
    out_num = data_expanded.to_numpy()
    out_str = data_expanded.columns.to_numpy()
    in_num =  in_feature.to_numpy()
    in_str =  in_feature.columns.to_numpy()
   
    char_list = ['omeg','gam','ras','alpl','kT']
  # ---------------------- 1. 获取 in_str 所有包含的符号 ----------------------
    in_chars = set()
    for col in in_str:
        chars = get_contained_chars(col, char_list)
        in_chars.update(chars)

    # ---------------------- 2. 找出 out_str 中包含相同符号的列索引 ----------------------
    idx_list = []
    for idx, col in enumerate(out_str):
        out_col_chars = get_contained_chars(col, char_list)
        # 如果有交集 → 要删除
        if out_col_chars & in_chars:
            idx_list.append(idx)

    # ---------------------- 3. 删除列 ----------------------
    if idx_list:
        out_num = np.delete(out_num, idx_list, axis=1)
        out_str = np.delete(out_str, idx_list)
        
    # 将in_num,in_str加入到out_num,out_str
    out_num = np.hstack((out_num, in_num))  # 横向拼接
    out_str = np.hstack((out_str, in_str))
    
    #把out_str相同符号的列装进一个列表    
    grouped_index_dic = {}
    for idx, col_name in enumerate(out_str):  # 拿到 索引idx + 列名
        chars = get_contained_chars(col_name, char_list)
        key = tuple(sorted(chars)) if chars else "other"
        
        if key not in grouped_index_dic:
            grouped_index_dic[key] = []
        grouped_index_dic[key].append(idx)  # 存入【索引】，不是列名！
    grouped_index_list = list(grouped_index_dic.values())
    
    grouped_out_str = []
    for indices in grouped_index_list:
        grouped_out_str.append(out_str[indices])

    grouped_out_num = []
    for indices in grouped_index_list:
        grouped_out_num.append(out_num[:, indices]) 
    
    out_num = grouped_out_num
    out_str = grouped_out_str
    
    # 对out进行二元交叉扩展

    ## 组合
    # 返回所有组合的可能
    combinations = list(itertools.combinations(range(len(out_num)), 2))
    # 对每一种组合进行计算
    for combination in combinations:
        # 取出一个组合中对应特征的索引
        idx1, idx2 = combination
        # 计算组合特征
        num_c, str_c = combine(out_num[idx1], out_num[idx2],
                               out_str[idx1], out_str[idx2])
        # 将组合特征添加到结果中
        out_num.append(num_c)
        out_str.append(str_c)

    ## 整合数据
    out_num = np.hstack(out_num)
    out_str = np.hstack(out_str)
    out_data = pd.DataFrame(out_num, columns=out_str)

    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data
    

# -------------------------- 4. 循环迭代核心逻辑 --------------------------
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
        data_expanded = expand_next(data_expanded,best_feature)

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

