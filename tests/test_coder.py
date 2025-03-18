import os
import pytest
import pandas as pd
from cscoder.coder import CSCOder

# 定义测试数据路径
TEST_DATA_PATH = "data/testcases.csv"
FAILED_CASES_PATH= "out/failcases.csv"


# 读取测试数据
@pytest.fixture(scope="module")
def test_cases():
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df['expected_code'] = test_df['expected_code'].str.replace('-', '')
    test_df.dropna(subset=['expected_code'])
    return test_df


@pytest.fixture(scope="module")
def matcher():
    """初始化 CSCOder 实例"""
    return CSCOder()


def test_matching_accuracy(matcher, test_cases):
    """测试匹配的准确率"""
    # 执行匹配
    results_df = matcher.find_best_matches(
        test_cases['job_name'], top_n=1, batch_size=10, return_df=True
    )

    # 添加预期结果列
    results_df = pd.concat(
        [results_df, test_cases[['expected_code', 'expected_name', 'job_name']]], axis=1
    )

    # 计算匹配正确的行
    results_df["matched"] = results_df["matched_code"] == results_df["expected_code"]
    matched_count = results_df["matched"].sum()
    accuracy = matched_count / len(test_cases)

    # 输出整体匹配准确率
    print(f"\n匹配准确率: {accuracy:.2%} ({matched_count}/{len(test_cases)})")

    # 处理匹配失败的案例
    failed_cases = results_df[~results_df["matched"]]
    
    # 保存匹配失败案例
    if not failed_cases.empty:
        os.makedirs(os.path.dirname(FAILED_CASES_PATH), exist_ok=True)

        failed_cases.to_csv(FAILED_CASES_PATH, index=False, encoding="utf-8-sig")

    # 断言匹配准确率 >= 90%
    assert accuracy >= 0.9, f"匹配准确率 {accuracy:.2%} 低于 90%!"