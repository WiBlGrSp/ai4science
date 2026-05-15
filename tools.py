import numpy as np
import pandas as pd

def check_bad_columns(df):
    # 1. 检查 NaN
    nan_mask = df.isna()
    
    # 2. 检查 inf 和 -inf 🔥🔥🔥（这是你缺的）
    inf_mask = np.isinf(df)
    
    # 3. 合并：只要是 NaN 或 inf 或 -inf 都算坏值
    bad_mask = nan_mask | inf_mask

    # 4. 每列是否有坏值
    has_bad = bad_mask.any()

    # 5. 坏列名单
    bad_cols = df.columns[has_bad].tolist()
    bad_num = len(bad_cols)

    print(f"总特征列数: {len(df.columns)}")
    print(f"含 NaN/inf/-inf 的列数: {bad_num}")

    if bad_num > 0:
        print("\n问题特征列:")
        print(bad_cols)

        # 统计每列坏值数量
        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()
        total = nan_count + inf_count

        print("\n每列坏值数量 (NaN + inf/-inf):")
        print(total[total > 0].sort_values(ascending=False))

    return bad_cols

def get_contained_chars(col_name: str, char_list: list[str]) -> set:
    """
    输入一个列名，返回它包含的所有符号（集合）
    """
    col_name = str(col_name)
    return {char for char in char_list if char in col_name}