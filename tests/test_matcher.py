import pytest
from cscoder.matcher import CSCOder

# 测试数据集 (非结构职业名称, 预期匹配的标准职业名称)
TEST_DATA = [
    ("4s店销售总监(集团)", "市场营销专业人员"),
    ("电梯销售（九江）", "商品营业员"),
    ("人力BP", "人力资源管理专业人员"),
    ("客户经理", "客户服务管理员"),
    ("储备副店长+带薪培训+免费住宿", "连锁经营管理师"),
    ("证券经纪人", "证券期货服务师"),
    ("（安徽）区域销售经理", "市场营销专业人员"),
    ("湖北恩施宣恩高薪聘请美睫师/美甲师", "美甲师"),
    ("labview工程师", "自动控制工程技术人员S"),
]


@pytest.fixture
def matcher():
    """初始化 CSCOder 实例"""
    return CSCOder()


# @pytest.mark.parametrize("job_title, expected_name", TEST_DATA)
# def test_find_best_match(matcher, job_title, expected_name):
#     """测试 find_best_match 是否能正确匹配职业名称"""
#     matches = matcher.find_best_match(job_title, version="csco22")

#     assert matches, f"❌ {job_title} 未找到匹配结果！"

#     best_match = matches[0]
#     matched_name = best_match["csco_name"]

#     assert matched_name == expected_name, (
#         f"❌ {job_title} 预期匹配 {expected_name}，但匹配到 {matched_name}"
#     )


def test_find_best_match_invalid_input(matcher):
    """测试 find_best_match 处理 None 或空字符串"""
    assert matcher.find_best_match("") == [], "❌ 空字符串应该返回空列表！"
    assert matcher.find_best_match(None) == [], "❌ None 应该返回空列表！"
    assert matcher.find_best_match("   ") == [], "❌ 纯空格字符串应该返回空列表！"
