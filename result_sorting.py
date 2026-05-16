from simplify_str import simplify
from feature_information import FeatureInformation
import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND

# 返回排序后要显示的结果，返回最优的特征:pd.dataframe
def sort_result_and_best(num_in:ND, str_in:ND, r2:list, p:list, 
                loss:list, n:int, num=0)->tuple[list[FeatureInformation],pd.DataFrame]:
    """
    对特征评估结果进行排序，并返回TopN特征信息与最优特征
    按R2从高到低排序，简化特征名称，封装特征信息并返回

    参数
    ----------
    num_in : ND
        特征数值矩阵
    str_in : ND
        特征名称数组
    r2 : list
        各特征对应的R2分数
    p : list
        各特征对应的p值
    loss : list
        各特征对应的损失值
    n : int
        需要返回的Top特征数量
    num : int, 可选
        起始序号，用于多轮排序时编号连续，默认为0

    返回
    ----------
    list_out : list[FeatureInformation]
        排序后的TopN特征信息列表
    pd.DataFrame
        最优特征DataFrame（单列）
    """
    # 初始化简化后特征名称数组
    str_sp = np.empty(len(str_in), dtype=object)

    # 遍历所有特征名称并进行简化
    for index in range(len(str_in)):
        str_sp[index] = simplify(str_in[index])

    # 根据R2分数从高到低排序，获取排序索引映射
    sortlist = sorted(enumerate(r2), key=lambda x: x[1], reverse=True)
    # 提取排序后的原始索引序列
    idx_sort = [x[0] for x in sortlist] 

    # 根据排序索引重新排列所有结果
    str_sp = str_sp[idx_sort]
    num_in = num_in[:, idx_sort]
    r2_out = [r2[i] for i in idx_sort] 
    p_out = [p[i] for i in idx_sort]
    loss_out = [loss[i] for i in idx_sort]

    # 初始化特征信息输出列表
    list_out = []
    # 取前n个有效特征进行封装
    for index in range(min(n, len(str_sp))):
        # 封装特征信息到FeatureInformation对象
        list_out.append(FeatureInformation(index+1+num, 
            pd.DataFrame(num_in[:, index], columns=[str_sp[index]]),
            p_out[index], r2_out[index], loss_out[index]))
        # 更新特征综合评分
        list_out[index].update_score()
    
    # 提取排序第一的最优特征（使用原始未简化名称）
    num_best_feature = num_in[:,idx_sort[0]]
    str_best_feature = str_in[idx_sort[0]]
    return list_out,pd.DataFrame(num_best_feature,columns=[str_best_feature])

def sort_result(num_in:ND, str_in:ND, r2:list, p:list, 
                loss:list, n:int, num=0)->list[FeatureInformation]:
    # 初始化
    str_sp = np.empty(len(str_in), dtype=object)

    # 简化字符串
    for index in range(len(str_in)):
        str_sp[index] = simplify(str_in[index])

    # 根据r2排序
    sortlist = sorted(enumerate(r2), key=lambda x: x[1], reverse=True)
    
    # sortlist = sorted(
    #     enumerate(r2),
    #     key=lambda x: 
    #         # 如果是 NaN → 给极低分 -1e99，丢到最后
    #         -1e99 if (x[1] is None or np.isnan(x[1]) or np.isinf(x[1])) 
    #         # 正常分数 → 正常排序
    #         else x[1],
    #     reverse=True
    # )
    
    # 得到排序后的索引
    idx_sort = [x[0] for x in sortlist] 

    # 根据排序后的索引整理输出
    str_sp = str_sp[idx_sort]
    num_in = num_in[:, idx_sort]
    r2_out = [r2[i] for i in idx_sort] 
    p_out = [p[i] for i in idx_sort]
    loss_out = [loss[i] for i in idx_sort]

    list_out = []
    for index in range(min(n, len(str_sp))):
        # 将结果存入FeatureInformation类中
        list_out.append(FeatureInformation(index+1+num, 
            pd.DataFrame(num_in[:, index], columns=[str_sp[index]]),
            p_out[index], r2_out[index], loss_out[index]))
        # 更新分数
        list_out[index].update_score()

    return list_out