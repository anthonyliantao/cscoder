import pytest
import time
import random
import pandas as pd
from cscoder.coder import CSCOder

@pytest.fixture(scope="module")
def coder():
    """CSCOder 实例 默认设置"""
    return CSCOder()

@pytest.fixture(scope="module")
def coder_no_cache():
    """CSCOder 实例 禁用缓存"""
    return CSCOder(disable_cache=True)

@pytest.fixture
def fake_cases():
    """测试数据：50 个不同的职业，复制 100 份并打乱顺序"""
    unique_jobs = [
        "医生", "工程师", "教师", "律师", "会计", "程序员", "护士", "厨师", "警察", "作家",
        "画家", "音乐家", "演员", "记者", "建筑师", "翻译", "研究员", "数据分析师", "市场经理", "销售员",
        "产品经理", "财务总监", "公关专员", "行政助理", "心理咨询师", "社工", "摄影师", "健身教练", "设计师", "农民",
        "电工", "司机", "理发师", "保安", "快递员", "超市收银员", "导游", "企业顾问", "保险代理人", "软件测试工程师",
        "儿童心理学家", "运动员", "银行职员", "信贷经理", "公务员", "研究生导师", "科学家", "木匠", "珠宝鉴定师", "化学家", "生物学家"
    ]
    job_list = unique_jobs * 100
    random.shuffle(job_list)
    return job_list

# @pytest.fixture(scope="module")
# def test_cases():
#     """测试案例集：外部文件导入"""
#     test_df = pd.read_csv('tests/data/testcases.csv')
#     test_df['expected_code'] = test_df['expected_code'].str.replace('-', '')
#     test_df.dropna(subset=['expected_code'])
#     return test_df
        
def test_cache_consistency(coder, coder_no_cache, fake_cases):
    """测试：启用缓存和不启用缓存，输出结果完全一致"""
    result = coder.find_best_matches(fake_cases, return_df=False)
    result_no_cache = coder_no_cache.find_best_matches(fake_cases, return_df=False)
    assert len(result) == len(result_no_cache), "缓存启用前后，匹配结果数量不一致！"

    codes = [item["matched_code"] for item in result]
    codes_no_cache = [item["matched_code"] for item in result_no_cache]
    assert codes == codes_no_cache, "缓存启用前后，匹配的职业代码不一致！"

def test_cache_speedup(coder, coder_no_cache, fake_cases):
    """测试：启用缓存后，匹配速度提升 1.5 倍以上"""
    start_time = time.time()
    coder.find_best_matches(fake_cases)
    timeuse = time.time() - start_time

    # 测试禁用缓存耗时
    start_time = time.time()
    coder_no_cache.find_best_matches(fake_cases)  # 第一次运行，填充缓存
    timeuse_no_cache = time.time() - start_time

    print(f"🚀 With Cache: {timeuse:.4f}s | No Cache: {timeuse_no_cache:.4f}s")

    # 第二次调用缓存，测试加速效果
    start_time = time.time()
    coder.find_best_matches(fake_cases)
    timeuse_2nd = time.time() - start_time

    print(f"🚀 Second With Cache: {timeuse_2nd:.4f}s")

    # 断言：第二次调用缓存时，比不使用缓存快 2 倍以上
    assert timeuse_2nd < timeuse_no_cache / 1.5, "缓存未显著提升速度！"
    
# def test_matching_accuracy(matcher, test_cases):
#     """测试匹配的准确率"""
#     # 执行匹配
#     results_df = matcher.find_best_matches(
#         test_cases['job_name'], top_n=1, batch_size=10, return_df=True
#     )

#     # 添加预期结果列
#     results_df = pd.concat(
#         [results_df, test_cases[['expected_code', 'expected_name', 'job_name']]], axis=1
#     )

#     # 计算匹配正确的行
#     results_df["matched"] = results_df["matched_code"] == results_df["expected_code"]
#     matched_count = results_df["matched"].sum()
#     accuracy = matched_count / len(test_cases)

#     # 输出整体匹配准确率
#     print(f"\n匹配准确率: {accuracy:.2%} ({matched_count}/{len(test_cases)})")

#     # 处理匹配失败的案例
#     failed_cases = results_df[~results_df["matched"]]
    
#     # 保存匹配失败案例
#     if not failed_cases.empty:
#         failed_cases.to_csv('tests/data/failcases.csv', index=False, encoding="utf-8-sig")

#     # 断言匹配准确率 >= 90%
#     assert accuracy >= 0.9, f"匹配准确率 {accuracy:.2%} 低于 90%!"