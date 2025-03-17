from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from .text_cleaner import clean_job_name


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


class CSCOder:
    def __init__(self, version="csco22", model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.version = version
        self._model = None
        self._csco_data = None
        self._alias_data = None

    @property
    def model(self):
        """加载 SentenceTransformer模型"""
        if self._model is None:
            print(f"Loading model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def csco_data(self):
        """加载 CSCO 数据"""
        if self._csco_data is None:
            file_path = os.path.join(DATA_DIR, f"{self.version}.csv")
            df = pd.read_csv(file_path, usecols=["code_num", "name"], dtype={"code_num": str})
            self._csco_data = df.set_index("code_num")["name"].to_dict()
        return self._csco_data

    @property
    def alias_data(self):
        """加载职业别名数据"""
        if self._alias_data is None:
            file_path = os.path.join(DATA_DIR, f"{self.version}_aliases.csv")
            df = pd.read_csv(file_path, usecols=["alias", "csco_code_num"])

            df["alias"] = df["alias"].str.strip().str.lower()
            df = df.drop_duplicates(subset=["alias"])

            embeddings = self._encode_texts(
                df["alias"].tolist(), show_progress_bar=True)

            self._alias_data = (df, embeddings)

        return self._alias_data

    @property
    def alias_df(self):
        """别名数据"""
        return self.alias_data[0]

    @property
    def alias_embeddings(self):
        """别名的向量表示"""
        return self.alias_data[1]

    def _encode_texts(self, texts, *args, **kwargs):
        """文本编码为向量"""
        return np.array(self.model.encode(texts, convert_to_numpy=True, *args, **kwargs))

    def _hierarchy_match(self, job_embeddings, top_n=1):
        """层级匹配"""
        similarity_matrix = 1 - \
            cdist(job_embeddings, self.alias_embeddings, metric="cosine")

        results = []
        for scores in similarity_matrix:
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:top_n]
            for idx in top_indices:
                sim_score = scores[idx]
                csco_code = self.alias_df.iloc[idx]["csco_code_num"]
                final_csco_code = self._check_code_for_similarity(
                    csco_code, sim_score)
                final_csco_name = self.csco_data.get(str(final_csco_code))
                results.append(
                    {
                    "csco_code": final_csco_code,
                    "csco_name": final_csco_name,
                    "similarity": sim_score
                    })

        return results

    def _check_code_for_similarity(self, csco_code, sim_score):
        """根据相似度返回对应层次的代码"""
        csco_code = str(csco_code)
        if sim_score >= 0.9:
            return csco_code                    # 7位代码
        elif sim_score >= 0.7:
            return csco_code[:5] + "00"          # 5位代码
        elif sim_score >= 0.5:
            return csco_code[:3] + "0000"  # 3位代码
        elif sim_score >= 0.3:
            return csco_code[:3] + "0000"    # 1位代码
        else:
           return None                     # 低于 0.3，不匹配

    def find_best_match(self, job_name, top_n=1, return_df=True):
        """匹配单个职业"""
        if not job_name.strip():
            return []
        
        job_embedding = self._encode_texts([clean_job_name(job_name)])
        match_results = self._hierarchy_match(job_embedding, top_n)
        results = [{"input": job_name, **match} for match in match_results]
        
        return results if not return_df else pd.DataFrame(results)

    def find_best_matches(self, job_names, top_n=1, batch_size=1000, return_df=True, show_progress=False):
        """匹配多个职业"""
        if isinstance(job_names, str):
            return self.find_best_match(job_names, top_n, return_df)

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
                job_embeddings = self._encode_texts(batch)
                batch_results = self._hierarchy_match(job_embeddings, top_n)
                results.extend([{"input": input, **match} for input, match in zip(batch, batch_results)])
                pbar.update(1)

        return pd.DataFrame(results) if return_df else results


if __name__ == "__main__":
    coder = CSCOder()
    
    # 匹配单个职业
    result = coder.find_best_match("软件工程师")
    print(result)

    # 匹配多个职业
    job_list = ["数据分析师", "产品经理", "注册会计师", "Java工程师"]
    result = coder.find_best_matches(job_list)
    print(result)
