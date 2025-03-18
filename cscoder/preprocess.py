import pandas as pd
import re


# 加载地理实体名词
def load_geo_ents(geo_file='data/cncity.csv'):
    """加载地理实体名称（省、市、区、县）"""
    geo_df = pd.read_csv(geo_file, usecols=['name', 'short_name'])
    geo_ents = set(geo_df.dropna().values.flatten())
    return geo_ents


GEO_ENTS = load_geo_ents()

# 加载停止词
with open("data/stopwords.txt", "r", encoding="utf-8") as f:
    STOPWORDS = {line.strip() for line in f if line.strip()}

def remove_puncs(text):
    """移除标点符号"""
    return re.compile(r"[^\w\s]").sub("", text)


def remove_dynamic_stopwords(text):
    """移除动态的停止词，如上X休X、早X晚X、月入X等"""
    pattern = re.compile(
        r'(提供|\+|有|包|管|\/)[吃住社保饭补餐补免费住宿宿舍分红带薪培训法休师傅带教五险一金员工餐]+|'  # 提供/+/有 + 福利
        r'(接受)?[小白无经验生熟手]+[均皆都]?可|'  # 接受小白、生熟手均可等
        r'月(入|入过|均过)?\d+[kK千万起]?|'  # 月入X、月过X等
        r'(薪资|待遇)面议|'  # X面议
        r'(年薪|薪资|保障薪资|底薪|无责)\d+[kK千万亿]?|'  # 保障薪资X、底薪X、无责X
        r'\d+[kK千万]?(年薪|薪资|保障薪资|底薪|无责)|'  # X保障薪资、X底薪、X无责
        r'\d+[kK]?(-|到)\d+[kK]?|'  # X-Y
        r'\d+[kK]?(\/)?(天|月|一天|每月)|'  # X/天、X/月
        r'\d+[kK千万]|'  # Xk、X千、X万
        r'早[零一二三四五六七八九十百千万\d]+晚[零一二三四五六七八九十百千万\d]+|'  # 早X晚Y
        r'[零一二三四五六七八九十\d]点(上|下)班|'  # X点下班
        r'上[零一二三四五六七八九十百千万几\d]+休[零一二三四五六七八九十百千万几\d]+|'  # 上X休Y
        r'月休[零一二三四五六七八九十\d]天|'  # 月休X天
        r'(周末)?[单双法]休|'
    )
    return pattern.sub("", text)


def remove_stopwords_from_file(text):
    """读取停止词文件中规定的停止词"""
    pattern = re.compile('|'.join(map(re.escape, STOPWORDS)))
    return pattern.sub("", text)


def remove_recruitment_verb(text):
    """移除 '招聘'，但保留 '招聘专员' 等白名单词"""
    whitelist = {"招聘专员", "招聘师"}
    return text if any(w in text for w in whitelist) else text.replace("招聘", "")


def remove_codelike_words(text):
    """
    移除字母开头 + 数字的部分（如 J10050），但保留白名单中的词（如 3D、UE4）。
    """
    whitelist = {"3d", "ue4", "a1", "a2",  "b1", "b2", "c1", "c2"}
    pattern = re.compile(r'([a-zA-Z]+)(\d+)([a-zA-Z]?)+')

    def replace_match(match):
        word = match.group(0)  # 获取整个匹配项
        return "" if word.lower() not in whitelist else word  # 保留白名单词
    
    return pattern.sub(replace_match, text)


def remove_geo_ents(text):
    """删除地理实体"""
    pattern = re.compile(r'(' + '|'.join(map(re.escape, GEO_ENTS)) + r')')
    return pattern.sub("", text)


def clean_job_name(text):
    """清理职位名称"""
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = remove_dynamic_stopwords(text)
    text = remove_stopwords_from_file(text)
    text = remove_recruitment_verb(text)
    text = remove_codelike_words(text)
    text = remove_geo_ents(text)
    text = remove_puncs(text)
    return text.strip()


if __name__ == "__main__":
    text = "昆明木木夕木目心...招聘陈列员"
    print(clean_job_name(text))