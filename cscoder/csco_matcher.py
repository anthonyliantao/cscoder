from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from tqdm import tqdm

from .data_loader import load_csco_aliases
from .text_cleaner import clean_job_name


class CSCOder:
    def __init__(self, version="csco22", model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.version = version
        self._model = None
        self._alias_cache = {}
    
    @property
    def model(self):
        """加载 SentenceTransformer模型"""
        if self._model is None:
            print(f"Loading model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_texts(self, texts, *args, **kwargs):
        """文本编码为向量"""
        return np.array(self.model.encode(texts, convert_to_numpy=True, *args, **kwargs))

    def get_alias_embeddings(self):
        """获取 CSCO 版本对应的职业别名库"""
        if self.version not in self._alias_cache:
            df = load_csco_aliases(self.version)
            embeddings = self.encode_texts(df["alias"].tolist(), show_progress_bar=True)
            self._alias_cache[self.version] = (df, embeddings)
        return self._alias_cache[self.version]
    
    def _calculate_alias_similarity(self, job_embeddings, top_n=1, threshold=0.5):
        """计算输入向量与职业别名库的相似度并筛选匹配结果"""
        alias_df, alias_embeddings = self.get_alias_embeddings()
        similarity_matrix = 1 - cdist(job_embeddings, alias_embeddings, metric="cosine")
        
        results = []
        for scores in similarity_matrix:
            sorted_indices = np.argsort(scores)[::-1]
            valid_indices = sorted_indices[scores[sorted_indices] >= threshold]
            top_indices = valid_indices[:top_n]
            results.append([
                {
                    "csco_code": alias_df.iloc[idx]["csco_code"],
                    "csco_name": alias_df.iloc[idx]["csco_name"],
                    "similarity": scores[idx]
                }for idx in top_indices
            ])

        return results

    def find_best_match(self, job_name, top_n=1, threshold=0.5):
        """匹配单个职业"""
        if not job_name.strip():
            return []
        job_embedding = self.encode_texts([clean_job_name(job_name)])
        return self._calculate_alias_similarity(job_embedding, top_n, threshold)[0]
            
    def find_best_matches(self, job_names, top_n=1, threshold=0.5, batch_size=1000, return_df=True, show_progress=False):
        """匹配多个职业"""
        if isinstance(job_names, str):
            return self.find_best_match(job_names, top_n, threshold)

        if isinstance(job_names, pd.Series):
            job_names = job_names.astype(str).tolist()
        
        if not isinstance(job_names, list):
            raise ValueError("job_names 必须是字符串、列表或 pd.Series")
        
        job_names = list(map(clean_job_name, job_names))
        total_jobs = len(job_names)
        batches = [job_names[i: i + batch_size] for i in range(0, total_jobs, batch_size)]
        results = []
        
        with tqdm(total=len(batches), desc="Processing Batches", unit="batch", disable=not show_progress) as pbar:
            for batch in batches:
                job_embeddings = self.encode_texts(batch)
                batch_results = self._calculate_alias_similarity(job_embeddings, top_n, threshold)
                results.extend(
                    {"input": job, **match} for job, matches in zip(batch, batch_results) for match in matches
                )
                pbar.update(1)

        return pd.DataFrame(results) if return_df else results
    

if __name__ == "__main__":
    cscoder = CSCOder()

    # 匹配单个职业
    result = cscoder.find_best_match("软件工程师", top_n=3)
    print(result)

    # 匹配多个职业
    job_list = ["数据分析师", "产品经理", "注册会计师", "Java工程师"]
    df = cscoder.find_best_matches(job_list, top_n=2)
    print(df)