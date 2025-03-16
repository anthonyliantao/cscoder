import json
import pytest
import os
import pandas as pd
from csco_matcher.csco_matcher import CSCOder

# 定义测试数据路径
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "test_cases.json")

# 读取测试数据
@pytest.fixture(scope="module")
def test_cases():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def matcher():
    """初始化 CSCOder 实例"""
    return CSCOder()

def test_matching_accuracy(matcher, test_cases):
    """测试匹配的准确率是否达到90%"""
    # 将测试输入转换为 pd.Series
    job_series = pd.Series([case["input"] for case in test_cases])
    expected_results = {case["input"]: case["expected"][0]["csco_code"] for case in test_cases}

    # 获取匹配结果，输出格式为 DataFrame
    results_df = matcher.find_best_matches(job_series, top_n=1, batch_size=10, return_df=True)

    # 计算准确率
    matched_count = sum(
        results_df["csco_code"].eq(results_df["job_title"].map(expected_results))
    )
    accuracy = matched_count / len(test_cases)
    
    print(f"匹配准确率: {accuracy:.2%}")
    assert accuracy >= 0.9, f"匹配准确率 {accuracy:.2%} 低于 90%!"