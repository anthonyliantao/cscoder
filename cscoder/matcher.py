from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from .data_loader import load_csco_aliases

# 加载预训练模型
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

# 预缓存职业别名及其向量（减少重复计算）
alias_cache = {}


def encode_texts(texts):
    """对文本列表进行编码"""
    return model.encode(texts, convert_to_numpy=True)


def find_best_match(job_title, top_n=1, version="csco22"):
    """
    计算 job_title 与 CSCO 职业别名的余弦相似度，返回最佳匹配结果（使用 SciPy）。
    """
    df = load_csco_aliases(version)
    aliases = df["alias"].tolist()

    # 缓存职业别名向量，避免重复计算
    if version not in alias_cache:
        alias_cache[version] = encode_texts(aliases)

    # 计算输入职业的向量
    job_embedding = encode_texts([job_title])  # shape (1, dim)

    # 计算余弦相似度 (1 - cosine_distance)
    similarity_scores = 1 - \
        cdist(job_embedding, alias_cache[version], metric="cosine")[0]

    # 获取最相似的 top_n 个别名
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "csco_code": df.iloc[idx]["csco_code"],
            "csco_name": df.iloc[idx]["csco_name"],
            "alias": df.iloc[idx]["alias"],
            "similarity": similarity_scores[idx]
        })

    return results


# 测试代码
if __name__ == "__main__":
    test_job = "软件工程师"
    matches = find_best_match(test_job, top_n=3)
    print(matches)
