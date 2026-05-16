import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND
from itertools import product
from tools import get_contained_chars
import numpy as np


def n_way_cross_expansion(out_num: list[np.ndarray], out_str: list[np.ndarray], N: int = 2):
    """
    N元交叉扩张函数
    对输入的多组数值特征与名称特征进行N元组合相乘，生成高阶组合特征
    
    参数
    ----------
    out_num : list[np.ndarray]
        待组合的数值特征列表，每个元素为一组特征矩阵
    out_str : list[np.ndarray]
        待组合的特征名称列表，与out_num一一对应
    N : int, 可选
        组合阶数，默认为2（二元交叉）
    
    返回
    ----------
    out_num : np.ndarray
        横向拼接后的所有原始特征 + 组合特征数值矩阵
    out_str : np.ndarray
        与数值矩阵对应的特征名称数组
    """
    # 生成所有N元特征组合索引
    combinations = list(itertools.combinations(range(len(out_num)), N))
    # 遍历每一种组合并计算组合特征
    for combination in combinations:
        # 调用combine函数生成组合特征与对应名称
        num_c, str_c = combine([out_num[idx] for idx in combination],
                               [out_str[idx] for idx in combination])
        # 将生成的组合特征追加到原特征列表
        out_num.append(num_c)
        out_str.append(str_c)

    # 横向拼接所有数值特征与名称
    out_num = np.hstack(out_num)
    out_str = np.hstack(out_str)
    return out_num, out_str


def univariate_transform(in_num, in_str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    单变量变换封装函数
    对每一列特征执行：平方根、平方、立方根、立方、倒数变换，扩充特征空间
    
    参数
    ----------
    in_num : 输入数值特征（分组后列表格式）
    in_str : 输入特征名称（与in_num对应）
    
    返回
    ----------
    out_num : 变换扩充后的数值特征
    out_str : 变换扩充后的特征名称
    """
    out_num = in_num
    out_str = in_str

    # 遍历每组特征，逐列执行幂次变换
    for index, name in enumerate(out_str):
        # 依次计算平方根、平方、立方根、立方变换结果
        num_p1, str_p1 = power(out_num[index], name, 1/2)
        num_p2, str_p2 = power(out_num[index], name, 2)
        num_p3, str_p3 = power(out_num[index], name, 1/3)
        num_p4, str_p4 = power(out_num[index], name, 3)
        # 横向拼接原始特征与四种幂次变换特征
        out_num[index] = np.hstack((out_num[index], num_p1, 
                                    num_p2, num_p3, num_p4))
        out_str[index] = np.hstack((out_str[index], str_p1,
                                    str_p2, str_p3, str_p4))
        
        # 计算倒数变换结果
        num_b, str_b = power(out_num[index], out_str[index], -1)
        # 横向拼接倒数变换结果
        out_num[index] = np.hstack((out_num[index], num_b))
        out_str[index] = np.hstack((out_str[index], str_b))

    return out_num, out_str


def remove_same_feature(data: pd.DataFrame, in_feature: pd.DataFrame, char_list) -> pd.DataFrame:
    """
    移除含指定符号的重复特征
    根据特征名称中包含的物理符号，删除data中与in_feature含相同符号的特征
    
    参数
    ----------
    data : 待过滤的数据集
    in_feature : 参考特征集（用于提取禁止出现的符号）
    char_list : 符号列表（如['omeg','gam']等）
    
    返回
    ----------
    过滤后的DataFrame
    """
    # 将DataFrame转换为numpy数组
    out_num = data.to_numpy()
    out_str = data.columns.to_numpy()
    in_num = in_feature.to_numpy()
    in_str = in_feature.columns.to_numpy()
    
    # 提取参考特征中包含的所有目标符号
    in_chars = set()
    for col in in_str:
        chars = get_contained_chars(col, char_list)
        in_chars.update(chars)

    # 筛选出包含相同符号的特征列索引
    idx_list = []
    for idx, col in enumerate(out_str):
        out_col_chars = get_contained_chars(col, char_list)
        # 特征符号存在交集则标记为待删除
        if out_col_chars & in_chars:
            idx_list.append(idx)

    # 删除指定索引的特征列
    if idx_list:
        out_num = np.delete(out_num, idx_list, axis=1)
        out_str = np.delete(out_str, idx_list)
    return pd.DataFrame(out_num, columns=out_str)


def split(in_num, in_str, char_list):
    """
    按特征名称中的符号对特征进行分组
    相同符号组合的特征分为一组，用于后续分组变换
    
    返回
    ----------
    out_num : 分组后的数值特征列表
    out_str : 分组后的特征名称列表
    """
    # 定义字典存储相同符号组合的特征索引
    grouped_index_dic = {}
    # 遍历所有特征，获取索引与列名
    for idx, col_name in enumerate(in_str):
        chars = get_contained_chars(col_name, char_list)
        # 以排序后的符号元组作为分组键
        key = tuple(sorted(chars)) if chars else "other"
        
        if key not in grouped_index_dic:
            grouped_index_dic[key] = []
        # 存储特征索引
        grouped_index_dic[key].append(idx)
    
    # 将字典值转换为列表格式
    grouped_index_list = list(grouped_index_dic.values())
    
    # 根据索引分组特征名称
    out_str = []
    for indices in grouped_index_list:
        out_str.append(in_str[indices])

    # 根据索引分组数值特征
    out_num = []
    for indices in grouped_index_list:
        out_num.append(in_num[:, indices]) 
    
    return out_num, out_str


def expand_next(data: pd.DataFrame, in_feature: pd.DataFrame) -> pd.DataFrame:
    """
    迭代式特征空间扩展
    1. 删除与最优特征重复符号的原始特征
    2. 合并最优特征到原始空间
    3. 按符号分组 → 单变量变换 → 二元交叉
    4. 清理异常值与重复列
    
    返回
    ----------
    扩展并清洗后的特征空间
    """
    # 定义物理符号列表，用于特征过滤与分组
    char_list = ['omeg', 'gam', 'ras', 'alpl', 'kT']
    # 删除原始空间中与最优特征符号重复的列
    data = remove_same_feature(data, in_feature, char_list)
    
    # 将数据转换为numpy数组格式
    out_num = data.to_numpy()
    out_str = data.columns.to_numpy()
    in_num = in_feature.to_numpy()
    in_str = in_feature.columns.to_numpy()

    # 横向拼接最优特征与原始特征
    out_num = np.hstack((out_num, in_num))
    out_str = np.hstack((out_str, in_str))
    
    # 按符号对特征空间进行分组
    out_num, out_str = split(out_num, out_str, char_list)
    
    # 对分组特征执行单变量变换
    out_num, out_str = univariate_transform(out_num, out_str)

    # 执行二元交叉扩张
    out_num, out_str = n_way_cross_expansion(out_num, out_str, 2)

    # 构建扩展后的DataFrame
    out_data = pd.DataFrame(out_num, columns=out_str)

    # 删除包含NaN或inf的异常列
    out_data = out_data.loc[:, ~(out_data.isna().any() | out_data.apply(np.isinf).any())]
    # 删除列名重复的特征
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data


def expand(ori_data: pd.DataFrame, N=2):
    '''
    全局特征空间N元扩展（一次性全空间扩张）
    对原始数据执行：分割 → 单变量变换 → N元交叉组合 → 数据清洗
    
    参数
    ----------
    ori_data : 原始输入数据集
    N : 扩展阶数，默认2
    
    返回
    ----------
    扩展完成的干净数据集
    '''
    # 复制原始数据并转换为numpy数组
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()
    
    # 按列分割数据与特征名称
    out_num = np.hsplit(out_num, out_num.shape[1])
    out_str = np.hsplit(out_str, out_str.shape[0])
    
    # 执行单变量变换
    out_num, out_str = univariate_transform(out_num, out_str)
    
    # 执行N元交叉扩张
    out_num, out_str = n_way_cross_expansion(out_num, out_str, N)

    # 构建扩展后的DataFrame
    out_data = pd.DataFrame(out_num, columns=out_str)

    # 删除包含异常值的列
    out_data = out_data.loc[:, ~(out_data.isna().any() | out_data.apply(np.isinf).any())]
    # 删除重复列名特征
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data


def power(in_num: ND, in_name: ND, p):
    '''
    幂次变换工具函数
    对输入特征执行指定次幂运算，并生成规范的特征名称
    
    参数
    ----------
    in_num : 输入数值
    in_name : 输入特征名
    p : 幂次
    
    返回
    ----------
    变换后数值 + 新特征名称
    '''
    # 一维数组转为二维，保证格式统一
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)

    # 执行幂运算
    out_num = np.power(in_num, p)
    # 生成新特征名称
    out_name = [f"(({n})**{p})" for n in in_name]

    return out_num, out_name


def combine(in_num_list: list[ND], in_name_list: list[ND]):
    '''
    N元特征组合函数（多元相乘）
    对多组特征进行全组合乘积，生成所有两两/多元相乘的组合特征
    
    参数
    ----------
    in_num_list : 多组输入数值特征
    in_name_list : 多组输入特征名称
    
    返回
    ----------
    组合后的数值矩阵 + 组合名称列表
    '''
    # 统一将一维数组转为二维
    for i in range(len(in_num_list)):
        if len(in_num_list[i].shape) == 1:
            in_num_list[i] = in_num_list[i].reshape(-1, 1)
    # 获取每组特征的列维度
    dims = [x.shape[1] for x in in_num_list]
    
    # 获取样本数量与总组合特征数
    num_datapoints = in_num_list[0].shape[0]
    num_featurevars = np.prod(dims)
    
    # 生成所有维度的组合索引
    index_combinations = product(*[range(d) for d in dims])

    # 初始化输出特征矩阵与名称列表
    out_num = np.ones((num_datapoints, num_featurevars))
    out_name = [''] * num_featurevars
    
    # 遍历所有组合并计算乘积特征
    idx = 0
    for indices in index_combinations:
        # 遍历组合内所有特征并累乘
        for i, idx_dim in enumerate(indices):
            out_num[:, idx] *= in_num_list[i][:, idx_dim]
            # 拼接组合特征名称
            if i == 0:
                out_name[idx] = in_name_list[0][idx_dim]
            else:
                out_name[idx] += f"*{in_name_list[i][idx_dim]}"
        idx += 1

    return out_num, out_name