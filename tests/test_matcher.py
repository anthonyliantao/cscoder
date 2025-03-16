import pytest
from cscoder.matcher import CSCOder

# 测试数据集
TEST_DATA = [
    ("4s店销售总监(集团)", "2-06-07-02", "市场营销专业人员"),
    ("电梯销售（九江）", "4-01-02-03", "商品营业员"),
    ("人力BP", "2-06-08-01", "人力资源管理专业人员"),
    ("客户经理", "4-07-02-03", "客户服务管理员"),
    ("储备副店长+带薪培训+免费住宿", "4-01-02-06", "连锁经营管理师"),
    ("证券经纪人", "4-05-02-01", "证券期货服务师"),
    ("（安徽）区域销售经理", "2-06-07-02", "市场营销专业人员"),
    ("湖北恩施宣恩高薪聘请美睫师/美甲师", "4-10-03-03", "美甲师"),
    ("labview工程师", "2-02-07-07", "自动控制工程技术人员S"),
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