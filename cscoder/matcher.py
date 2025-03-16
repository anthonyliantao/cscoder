from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
from .data_loader import load_csco_aliases


class CSCOder:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.alias_cache = {}

    def load_model(self):
        """加载 SentenceTransformer模型"""
        if self.model is None:
            print(f"Loading model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts):
        """对文本列表进行编码"""
        self.load_model()
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def get_alias_embeddings(self, version):
        """获取 CSCO 版本对应的职业别名编码"""
        if version not in self.alias_cache:
            df = load_csco_aliases(version)
            aliases = df["alias"].tolist()
            self.alias_cache[version] = {
                "df": df,  # 直接存 DataFrame，减少多次读取 CSV
                "embeddings": self.encode_texts(aliases)
            }
        return self.alias_cache[version]["df"], self.alias_cache[version]["embeddings"]

    def find_best_match(self, job_title, top_n=1, version="csco22", threshold=0.5):
        """单个职业匹配"""
        if not job_title or not job_title.strip():
            return []

        df, alias_embeddings = self.get_alias_embeddings(version)
        job_embedding = self.encode_texts([job_title])

        similarity_scores = 1 - \
            cdist(job_embedding, alias_embeddings, metric="cosine")[0]

        valid_indices = np.where(similarity_scores >= threshold)[0]
        sorted_indices = valid_indices[np.argsort(
            similarity_scores[valid_indices])[::-1]]
        if top_n:
            sorted_indices = sorted_indices[:top_n]

        return [{"csco_code": df.iloc[idx]["csco_code"],
                 "csco_name": df.iloc[idx]["csco_name"],
                 "similarity": similarity_scores[idx]} for idx in sorted_indices]

    def find_best_matches(self, job_titles, top_n=1, version="csco22", threshold=0.5):
        """匹配多个职业"""
        df, alias_embeddings = self.get_alias_embeddings(version)
        job_embeddings = self.encode_texts(job_titles)

        similarity_matrix = 1 - \
            cdist(job_embeddings, alias_embeddings, metric="cosine")

        results = []
        for i, job_title in enumerate(job_titles):
            similarity_scores = similarity_matrix[i]
            valid_indices = np.where(similarity_scores >= threshold)[0]
            sorted_indices = valid_indices[np.argsort(
                similarity_scores[valid_indices])[::-1]]
            if top_n:
                sorted_indices = sorted_indices[:top_n]

            job_results = [{"csco_code": df.iloc[idx]["csco_code"],
                            "csco_name": df.iloc[idx]["csco_name"],
                            "alias": df.iloc[idx]["alias"],
                            "similarity": similarity_scores[idx]} for idx in sorted_indices]

            results.append({"job_title": job_title, "matches": job_results})

        return results
