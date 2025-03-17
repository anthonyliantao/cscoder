from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil

from .data_loader import load_csco_aliases
from .text_cleaner import clean_job_name


class CSCOder:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.alias_cache = {}

    def load_model(self):
        """加载 SentenceTransformer模型"""
        if self.model is None:
            print(f"Loading model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts, *args, **kwargs):
        """对文本列表进行编码"""
        self.load_model()
        return np.array(self.model.encode(texts, convert_to_numpy=True, *args, **kwargs))

    def get_alias_embeddings(self, version):
        """获取 CSCO 版本对应的职业别名编码"""
        if version not in self.alias_cache:
            df = load_csco_aliases(version)
            aliases = df["alias"].tolist()
            self.alias_cache[version] = {
                "df": df,  # 直接存 DataFrame，减少多次读取 CSV
                "embeddings": self.encode_texts(aliases, show_progress_bar=True)
            }
        return self.alias_cache[version]["df"], self.alias_cache[version]["embeddings"]

    def find_best_match(self, job_name, top_n=1, version="csco22", threshold=0.5):
        """匹配单个职业"""
        if not job_name or not job_name.strip():
            return []
        
        job_name = clean_job_name(job_name)

        df, alias_embeddings = self.get_alias_embeddings(version)
        job_embedding = self.encode_texts([job_name])

        similarity_scores = 1 - cdist(job_embedding, alias_embeddings, metric="cosine")[0]

        valid_indices = np.where(similarity_scores >= threshold)[0]
        sorted_indices = valid_indices[np.argsort(
            similarity_scores[valid_indices])[::-1]]
        if top_n:
            sorted_indices = sorted_indices[:top_n]

        return [{"csco_code": df.iloc[idx]["csco_code"],
                 "csco_name": df.iloc[idx]["csco_name"],
                 "similarity": similarity_scores[idx]} for idx in sorted_indices]
    
    def find_best_matches(self, job_names, top_n=1, version="csco22", threshold=0.5, batch_size=1000, return_df=True, show_progress=True):
        """匹配多个职业"""
        if job_names is None:
            return []
        
        if isinstance(job_names, str):
            return self.find_best_match(job_names, top_n, version, threshold)
        
        if isinstance(job_names, pd.Series):
            job_names = job_names.astype(str).tolist()
        
        if not isinstance(job_names, list):
            raise ValueError("job_titles 必须是字符串、列表或pd.Series")
        
        alias_df, alias_embeddings = self.get_alias_embeddings(version)
        
        job_names = [clean_job_name(job_name) for job_name in job_names]

        total_jobs = len(job_names)
        num_batches = total_jobs / batch_size
        batches = [job_names[i: i + batch_size] for i in range(0, total_jobs, batch_size)]
        
        results = []
        with tqdm(total=num_batches, desc="Processing Batches", unit="batch", disable=not show_progress) as pbar:
            for batch in batches:
                job_embeddings = self.encode_texts(batch)
                similarity_matrix = 1 - cdist(job_embeddings, alias_embeddings, metric="cosine")

                for i, job_title in enumerate(batch):
                    similarity_scores = similarity_matrix[i]
                    valid_indices = np.where(similarity_scores >= threshold)[0]
                    sorted_indices = valid_indices[np.argsort(similarity_scores[valid_indices])[::-1]]
                    if top_n:
                        sorted_indices = sorted_indices[:top_n]

                    match_results = [{"input": job_title,
                                    "csco_code": alias_df.iloc[idx]["csco_code"],
                                    "csco_name": alias_df.iloc[idx]["csco_name"],
                                    "similarity": similarity_scores[idx]} for idx in sorted_indices]

                    results.extend(match_results)
                    
                pbar.update(1)
            
        return pd.DataFrame(results) if return_df else results