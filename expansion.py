import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND
from itertools import product
from tools import get_contained_chars
import numpy as np


def n_way_cross_expansion(out_num:list[np.ndarray],out_str:list[np.ndarray],N:int):
    """
    n元交叉扩张
    """
    ## 组合
    # 返回所有组合的可能
    combinations = list(itertools.combinations(range(len(out_num)), N))
    # 对每一种组合进行计算
    for combination in combinations:
        # 计算组合特征
        num_c, str_c = combine_n([out_num[idx] for idx in combination],
                               [out_str[idx] for idx in combination])
        # 将组合特征添加到结果中
        out_num.append(num_c)
        out_str.append(str_c)

    ## 整合数据
    out_num = np.hstack(out_num)
    out_str = np.hstack(out_str)
    return out_num,out_str
    
def univariate_transform(in_num, in_str)->tuple[list[np.ndarray],list[np.ndarray]]:
    """
    单变量代换封装函数（平方根、平方、立方根、立方、倒数）
    """
    
    out_num = np.hsplit(in_num, in_num.shape[1])
    out_str = np.hsplit(in_str, in_str.shape[0])
    
    for index, name in enumerate(out_str):
        
        num_p1, str_p1 = power(out_num[index], name, 1/2)
        num_p2, str_p2 = power(out_num[index], name, 2)
        num_p3, str_p3 = power(out_num[index], name, 1/3)
        num_p4, str_p4 = power(out_num[index], name, 3)
        out_num[index] = np.hstack((out_num[index], num_p1, 
                                    num_p2, num_p3, num_p4))
        out_str[index] = np.hstack((out_str[index], str_p1,
                                    str_p2, str_p3, str_p4))
        
        num_b, str_b = power(out_num[index], out_str[index], -1)
        out_num[index] = np.hstack((out_num[index], num_b))
        out_str[index] = np.hstack((out_str[index], str_b))

    return out_num, out_str
def remove_same_feature(data:pd.DataFrame,in_feature:pd.DataFrame,char_list)->pd.DataFrame:
    out_num = data.to_numpy()
    out_str = data.columns.to_numpy()
    in_num =  in_feature.to_numpy()
    in_str =  in_feature.columns.to_numpy()
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
    return pd.DataFrame(out_num,columns=out_str)

#分割符号空间
def split(in_num,in_str,char_list):
    
    #具有相同符号的位序放入字典
    grouped_index_dic = {}
    for idx, col_name in enumerate(in_str):  # 拿到 索引idx + 列名
        chars = get_contained_chars(col_name, char_list)
        key = tuple(sorted(chars)) if chars else "other"
        
        if key not in grouped_index_dic:
            grouped_index_dic[key] = []
        grouped_index_dic[key].append(idx)  # 存入【索引】，不是列名！
    grouped_index_list = list(grouped_index_dic.values())
    
    out_str = []
    for indices in grouped_index_list:
        out_str.append(in_str[indices])

    out_num = []
    for indices in grouped_index_list:
        out_num.append(in_num[:, indices]) 
    
    return out_num,out_str

#迭代扩展符号空间,将上一轮最优特征加入到原始空间，删去含重复符号的,获得扩展空间
def expand_next(data:pd.DataFrame,in_feature:pd.DataFrame)->pd.DataFrame:
    # 将原始空间中含有与infeature相同符号的列删除
    char_list = ['omeg','gam','ras','alpl','kT']
    data = remove_same_feature(data,in_feature,char_list)
    
    ## 初始化
    out_num = data.to_numpy()
    out_str = data.columns.to_numpy()
    in_num =  in_feature.to_numpy()
    in_str =  in_feature.columns.to_numpy()

    # 将最优特征加入到原始空间
    out_num = np.hstack((out_num, in_num))  # 横向拼接
    out_str = np.hstack((out_str, in_str))
    
    # 分割符号不同的空间
    out_num,out_str = split(out_num,out_str,char_list)

    # 对out进行二元交叉扩展
    out_num,out_str = n_way_cross_expansion(out_num,out_str,2)


    out_data = pd.DataFrame(out_num, columns=out_str)

    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data
    


def expand_n(ori_data:pd.DataFrame,N:int):
    '''
    对数据进行N元扩展
    '''
    ## 初始化
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()
    # 单变量代换
    out_num,out_str= univariate_transform(out_num,out_str)
    
    ## n元交叉扩张
    out_num,out_str = n_way_cross_expansion(out_num,out_str,N)

    out_data = pd.DataFrame(out_num, columns=out_str)

    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data
# 把x扩展为x的幂次的列表，在将x1,x2的幂次列表中元素两两组合构成二元交叉的幂次列表
# 最终输出初始特征集合 + 2个幂次扩展集合 + 二元交叉扩展集合
def expand(ori_data:pd.DataFrame):
    '''
    对数据进行扩展
    '''
    ## 初始化
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()
    # 单变量代换
    out_num,out_str= univariate_transform(out_num,out_str)

    ## 2元交叉扩张
    out_num,out_str = n_way_cross_expansion(out_num,out_str,2)
    
    out_data = pd.DataFrame(out_num, columns=out_str)
    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data

def power(in_num:ND, in_name:ND, p):
    '''
    对数据进行幂运算
    '''
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)

    out_num = np.power(in_num, p)
    out_name = [f"({n}**{p})" for n in in_name]

    return out_num, out_name
def combine_n(in_num_list:list[ND],in_name_list:list[ND]):
    '''
    对数据进行n元组合运算'
    '''
    for i in range(len(in_num_list)):
        if len(in_num_list[i].shape) == 1:
            in_num_list[i] = in_num_list[i].reshape(-1, 1)
    # 获取每个特征矩阵的 列数（特征维度）
    dims = [x.shape[1] for x in in_num_list]
    
    # 组合特征的数量
    
    num_datapoints = in_num_list[0].shape[0]
    num_featurevars = np.prod(dims)
    
    # 生成所有组合的索引（N 重循环核心！）
    index_combinations = product(*[range(d) for d in dims])

    # 初始化组合特征矩阵和名称列表
    out_num = np.ones((num_datapoints, num_featurevars))
    out_name = ['']*num_featurevars
    # 计算组合特征
    idx = 0
    # 遍历所有 N 重组合
    for indices in index_combinations:
        
        # 初始化当前组合值为 1
        val = np.ones(num_datapoints)
        
        # 依次相乘 N 个特征（自动适配 N 重）
        for i, idx_dim in enumerate(indices):
            out_num[:,idx] *= in_num_list[i][:, idx_dim]
            # 拼接名称（如：a*b*c）
            if i == 0:
                out_name[idx] = in_name_list[0][idx_dim]
            else:
                out_name[idx]+=f"*{in_name_list[i][idx_dim]}"
        idx += 1

    return out_num, out_name
def combine(in_num1:ND, in_num2:ND, in_name1:ND, in_name2:ND):
    '''
    对数据进行二元组合运算'
    '''
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)

    # 组合特征的数量
    num_datapoints = in_num1.shape[0]
    num_featurevars = in_num1.shape[1] * in_num2.shape[1]

    # 初始化组合特征矩阵和名称列表
    out_num = np.zeros((num_datapoints, num_featurevars))
    out_name = ['']*num_featurevars

    # 计算组合特征
    idx = 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            out_num[:, idx] = in_num1[:, i] * in_num2[:, j]
            out_name[idx] = f"({in_name1[i]}*{in_name2[j]})"
            idx += 1

    return out_num, out_name