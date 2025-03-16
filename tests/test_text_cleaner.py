import unittest
from cscoder.text_cleaner import clean_job_name

class TestTextCleaner(unittest.TestCase):

    def test_remove_dynamic_stopwords(self):
        self.assertEqual(clean_job_name("提供吃住社保"), "")
        self.assertEqual(clean_job_name("月入过万"), "")
        self.assertEqual(clean_job_name("薪资面议"), "")

    def test_remove_stopwords_from_file(self):
        self.assertEqual(clean_job_name("电话客服生熟手均可"), "电话客服")
        self.assertEqual(clean_job_name("销售助理轻松上手"), "销售助理")
        self.assertEqual(clean_job_name("打包工人长白班"), "打包工人")

    def test_remove_recruitment_verb(self):
        self.assertEqual(clean_job_name("招聘"), "")
        self.assertEqual(clean_job_name("招聘专员"), "招聘专员")

    def test_remove_codelike_words(self):
        self.assertEqual(clean_job_name("高级产品经理（KG0050）"), "高级产品经理") # 大写多字母+数字
        self.assertEqual(clean_job_name("服装设计师【k1039】"), "服装设计师")  # 小写字母+数字
        self.assertEqual(clean_job_name("动画设计（m601x)"), "动画设计")  # 字母+数字+字母
        self.assertEqual(clean_job_name("3D设计师"), "3D设计师")  # 保留白名单词
        self.assertEqual(clean_job_name("C1驾照司机"), "C1驾照司机")  # 保留白名单词

    def test_remove_geo_ents(self):
        self.assertEqual(clean_job_name("北京市"), "")
        self.assertEqual(clean_job_name("上海市"), "")

    def test_remove_puncs(self):
        self.assertEqual(clean_job_name("测试，测试。"), "测试测试")
        self.assertEqual(clean_job_name("你好！世界？"), "你好世界")

    def test_clean_job_name(self):
        self.assertEqual(clean_job_name("提供吃住社保，月入过万，有五险一金，招聘专员J10050，北京市"), "招聘专员")
        self.assertEqual(clean_job_name("薪资面议，3D设计师，base上海"), "3D设计师")

if __name__ == '__main__':
    unittest.main()
