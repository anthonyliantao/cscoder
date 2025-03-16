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
        """延迟加载 SentenceTransformer，避免启动时加载"""
        if self.model is None:
            print(f"Loading model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts):
        """对文本列表进行编码"""
        self.load_model()
        return self.model.encode(texts, convert_to_numpy=True)

    def find_best_match(self, job_title, top_n=1, version="csco22", threshold=0.5):
        """单个职业匹配"""
        self.load_model()
        df = load_csco_aliases(version)
        aliases = df["alias"].tolist()

        if version not in self.alias_cache:
            self.alias_cache[version] = self.encode_texts(aliases)

        job_embedding = self.encode_texts([job_title])
        similarity_scores = 1 - cdist(job_embedding, self.alias_cache[version], metric="cosine")[0]

        valid_indices = np.where(similarity_scores >= threshold)[0]
        sorted_indices = valid_indices[np.argsort(similarity_scores[valid_indices])[::-1]]
        if top_n:
            sorted_indices = sorted_indices[:top_n]

        results = [{"csco_code": df.iloc[idx]["csco_code"],
                    "csco_name": df.iloc[idx]["csco_name"],
                    "alias": df.iloc[idx]["alias"],
                    "similarity": similarity_scores[idx]} for idx in sorted_indices]

        return results

    def find_best_matches_batch(self, job_titles, top_n=1, version="csco22", threshold=0.5):
        """批量匹配多个职业"""
        self.load_model()
        df = load_csco_aliases(version)
        aliases = df["alias"].tolist()

        if version not in self.alias_cache:
            self.alias_cache[version] = self.encode_texts(aliases)

        job_embeddings = self.encode_texts(job_titles)
        similarity_matrix = 1 - cdist(job_embeddings, self.alias_cache[version], metric="cosine")

        results = []
        for i, job_title in enumerate(job_titles):
            similarity_scores = similarity_matrix[i]
            valid_indices = np.where(similarity_scores >= threshold)[0]
            sorted_indices = valid_indices[np.argsort(similarity_scores[valid_indices])[::-1]]
            if top_n:
                sorted_indices = sorted_indices[:top_n]

            job_results = [{"csco_code": df.iloc[idx]["csco_code"],
                            "csco_name": df.iloc[idx]["csco_name"],
                            "alias": df.iloc[idx]["alias"],
                            "similarity": similarity_scores[idx]} for idx in sorted_indices]

            results.append({"job_title": job_title, "matches": job_results})

        return results
