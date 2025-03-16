import unittest
from cscoder.text_cleaner import clean_job_name

class TestTextCleaner(unittest.TestCase):

    def test_remove_dynamic_stopwords(self):
        self.assertEqual(clean_job_name("提供吃住社保"), "")
        self.assertEqual(clean_job_name("月入过万"), "")
        self.assertEqual(clean_job_name("薪资面议"), "")

    def test_remove_stopwords_from_file(self):
        self.assertEqual(clean_job_name("这是一个测试"), "测试")
        self.assertEqual(clean_job_name("请忽略这些停止词"), "忽略停止词")

    def test_remove_recruitment_verb(self):
        self.assertEqual(clean_job_name("招聘"), "")
        self.assertEqual(clean_job_name("招聘专员"), "招聘专员")

    def test_remove_codelike_words(self):
        self.assertEqual(clean_job_name("J10050"), "")
        self.assertEqual(clean_job_name("3D设计师"), "3D设计师")

    def test_remove_geo_ents(self):
        self.assertEqual(clean_job_name("北京市"), "")
        self.assertEqual(clean_job_name("上海市"), "")

    def test_remove_puncs(self):
        self.assertEqual(clean_job_name("测试，测试。"), "测试测试")
        self.assertEqual(clean_job_name("你好！世界？"), "你好世界")

    def test_clean_job_name(self):
        self.assertEqual(clean_job_name("提供吃住社保，月入过万，招聘专员J10050，北京市"), "招聘专员")
        self.assertEqual(clean_job_name("薪资面议，3D设计师，上海市"), "3D设计师")

if __name__ == '__main__':
    unittest.main()
