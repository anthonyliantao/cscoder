import json
import pytest
import pandas as pd
from cscoder.csco_matcher import CSCOder

# 定义测试数据路径
TEST_DATA_PATH = "data/testcases.xlsx"
OUTPUT_DATA_PATH = "data/testcases_result.csv"

# 读取测试数据


@pytest.fixture(scope="module")
def test_cases():
    test_df = pd.read_excel(TEST_DATA_PATH)
    test_df['expected_code'] = test_df['expected_code'].str.replace('-', '')
    test_df.dropna(subset=['expected_code'])
    return test_df


@pytest.fixture(scope="module")
def matcher():
    """初始化 CSCOder 实例"""
    return CSCOder()


def test_matching_accuracy(matcher, test_cases):
    """测试匹配的准确率"""
    # 匹配职业名称列
    results_df = matcher.find_best_matches(
        test_cases['job_name'], top_n=1, batch_size=10, return_df=True)

    # 添加预期结果列
    results_df = pd.concat(
        [results_df, test_cases[['expected_code', 'expected_name']]], axis=1)

    # 输出匹配结果 方便调试
    results_df.to_csv(OUTPUT_DATA_PATH, index=False)

    matched_count = sum(results_df["csco_code"] == results_df["expected_code"])
    accuracy = matched_count / len(test_cases)

    print(f"匹配准确率: {accuracy:.2%}")
    assert accuracy >= 0.9, f"匹配准确率 {accuracy:.2%} 低于 90%!"
