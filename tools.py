import numpy as np
import pandas as pd


def get_contained_chars(col_name: str, char_list: list[str]) -> set:
    """
    输入一个列名，返回它包含的所有符号（集合）
    """
    col_name = str(col_name)
    return {char for char in char_list if char in col_name}