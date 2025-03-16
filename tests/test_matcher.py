import pytest
import pandas as pd
from unittest.mock import patch
import numpy as np

from cscoder.matcher import CSCOder


@pytest.fixture
def csco():
    """创建一个 CSCOder 实例 避免每个测试都重复加载模型"""
    csco = CSCOder()
    csco.load_model()
    return csco


# def test_single_string_input(csco):
#     """测试单个字符串输入"""
#     result = csco.find_best_matches("软件工程师", top_n=1)
#     assert isinstance(result, list)  # 返回列表
#     assert len(result) > 0  # 结果不为空
#     assert isinstance(result[0], dict)  # 结果中包含字典
#     assert set(["csco_code", "csco_name", "similarity"]).issubset(
#         result[0].keys())  # 字典包含指定键


# def test_list_input(csco):
#     """测试列表输入"""
#     job_list = ["软件工程师", "数据分析师", "市场经理"]
#     result = csco.find_best_matches(job_list, top_n=1)
#     assert isinstance(result, list)  # 返回列表
#     assert len(result) == len(job_list)   # 结果长度与输入相同
#     assert all(isinstance(item, dict) for item in result)  # 结果中包含字典
#     assert all(set(["csco_code", "csco_name", "similarity"]).issubset(
#         item.keys()) for item in result)  # 字典包含指定键


def test_pandas_series_input(csco):
    """测试 pandas.Series 输入"""
    job_series = pd.Series(["软件工程师", "", None, "  ", "数据分析师", 123, float("nan")])
    result = csco.find_best_matches(job_series, top_n=1)
    assert isinstance(result, list)  # 返回列表
    assert len(result) == len(job_series)  # 结果长度与输入相同


# def test_empty_input(csco):
#     """测试空输入"""
#     assert csco.find_best_matches("", top_n=1) == []
#     assert csco.find_best_matches([], top_n=1) == []
#     assert csco.find_best_matches(pd.Series([]), top_n=1) == []
#     assert csco.find_best_matches(None, top_n=1) == []


# def test_invalid_input(csco):
#     """测试无效输入"""
#     with pytest.raises(ValueError):
#         csco.find_best_matches(123, top_n=1)  # 非字符串/列表输入应报错
