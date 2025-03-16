import pytest
from cscoder.matcher import CSCOder

# 测试数据集
TEST_DATA = [
    ("数据分析师", "2512", "数据分析师"),
    ("软件工程师", "2513", "软件工程师"),
    ("市场营销经理", "1221", "市场营销经理"),
    ("产品经理", "2519", "产品经理"),
    ("AI 研究员", "2521", "人工智能工程师"),
]

@pytest.fixture
def matcher():
    """初始化 CSCOder 实例"""
    return CSCOder()

@pytest.mark.parametrize("job_title, expected_code, expected_name", TEST_DATA)
def test_find_best_match(matcher, job_title, expected_code, expected_name):
    """测试 find_best_match 函数"""
    matches = matcher.find_best_match(job_title, top_n=1, version="csco22")

    assert matches, f"{job_title} 未找到匹配结果！"

    best_match = matches[0]
    matched_code = best_match["csco_code"]
    matched_name = best_match["csco_name"]

    assert matched_code == expected_code, f"{job_title} 预期代码 {expected_code}，但匹配到 {matched_code}"
    assert matched_name == expected_name, f"{job_title} 预期名称 {expected_name}，但匹配到 {matched_name}"