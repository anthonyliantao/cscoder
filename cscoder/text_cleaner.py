import pandas as pd
import re


# 加载地理实体名词
def load_geo_ents(geo_file='data/cncity.xlsx'):
    """加载地理实体名称（省、市、区、县）"""
    geo_df = pd.read_excel(geo_file, usecols=['name', 'short_name'])
    geo_ents = set(geo_df.dropna().values.flatten())
    return geo_ents


GEO_ENTS = load_geo_ents()

# 加载停止词
STOPWORDS = set(open("data/stopwords.txt",
                encoding="utf-8").read().splitlines())


def remove_puncs(text):
    """移除标点符号"""
    return re.compile(r"[^\w\s]").sub("", text)


def remove_dynamic_stopwords(text):
    """移除动态的停止词，如上X休X、早X晚X、月入X等"""
    pattern = re.compile(
        r'(提供|\+|有|包|管)[吃住社保饭补餐补免费住宿宿舍分红带薪培训法休师傅带教]+|'  # 提供/+/有 + 福利
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
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, STOPWORDS)) + r')\b')
    return pattern.sub("", text)


def remove_recruitment_verb(text):
    """移除 '招聘'，但保留 '招聘专员' 等白名单词"""
    whitelist = {"招聘专员", "招聘师"}
    return text if any(w in text for w in whitelist) else text.replace("招聘", "")


def remove_codelike_words(text):
    """移除字母开头+数字的编码（如 J10050），但保留 3D、UE4 等有实际意义的白名单"""
    whitelist = {"3d", "ue4", "b2"}
    return re.sub(r'\b[a-zA-Z]+\d+\b', lambda m: m.group() if m.group().lower() in whitelist else "", text)


def remove_geo_ents(text):
    """删除地理实体"""
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, GEO_ENTS)) + r')\b')
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
