from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import load_csco
from .preprocess import clean_job_name


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
        """加载 CSCO 数据（code -> name）"""
        if self._csco_data is None:
            df = load_csco(self.version)
            self._csco_data = df.set_index("code")["name"].to_dict()
        return self._csco_data

    @property
    def alias_data(self):
        """加载职业别名数据"""
        if self._alias_data is None:
            df = load_csco(self.version)
            alias_list = []
            code_list = []

            for _, row in df.iterrows():
                for alias in row["alias"]:
                    alias_list.append(alias)
                    code_list.append(row["code"])

            alias_df = pd.DataFrame({"alias": alias_list, "code": code_list})
            embeddings = self._encode_texts(alias_list, show_progress_bar=True)

            self._alias_data = (alias_df, embeddings)
        return self._alias_data

    @property
    def alias_df(self):
        """职业别名数据"""
        return self.alias_data[0]

    @property
    def alias_embeddings(self):
        """职业别名向量表示"""
        return self.alias_data[1]

    def _encode_texts(self, texts, *args, **kwargs):
        """文本编码为向量"""
        return np.array(self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, *args, **kwargs))

    def _match(self, job_embeddings, top_n=1, match_prt_lvl=False):
        """相似度匹配"""
        similarity_matrix = 1 - \
            cdist(job_embeddings, self.alias_embeddings, metric="cosine")

        results = []
        for scores in similarity_matrix:
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:top_n]
            for idx in top_indices:
                sim_score = scores[idx]
                csco_code = self.alias_df.iloc[idx]["code"]
                csco_code = self._match_parent_level(csco_code, sim_score) if match_prt_lvl else csco_code
                csco_name = self.csco_data.get(str(csco_code))
                results.append(
                    {
                        "matched_code": csco_code,
                        "matched_name": csco_name,
                        "similarity": sim_score
                    })

        return results

    def _match_parent_level(self, csco_code, sim_score):
        """根据相似度返回对应层级的父级代码"""
        csco_code = str(csco_code)
        if sim_score >= 0.8:
            return csco_code                     # 7位代码
        elif sim_score >= 0.6:
            return csco_code[:5] + "00"          # 5位代码
        elif sim_score >= 0.4:
            return csco_code[:3] + "0000"        # 3位代码
        elif sim_score >= 0.2:
            return csco_code[:3] + "0000"        # 1位代码
        else:
            return "8000000"                      # 低于 0.2，返回不便分类人员

    def find_best_match(self, job_name, top_n=1, return_df=True, match_prt_level=False):
        """匹配单个职业"""
        if not job_name.strip():
            return []

        job_embedding = self._encode_texts([clean_job_name(job_name)])
        match_results = self._match(job_embedding, top_n, match_prt_level)
        results = [{"input": job_name, **match} for match in match_results]

        return results if not return_df else pd.DataFrame(results)

    def find_best_matches(self, job_names, top_n=1, batch_size=1000, return_df=True, show_progress=False, match_prt_level=False):
        """匹配多个职业"""
        if isinstance(job_names, str):
            return self.find_best_match(job_names, top_n, return_df, match_prt_level)

        if isinstance(job_names, pd.Series):
            job_names = job_names.astype(str).tolist()

        if not isinstance(job_names, list):
            raise ValueError("job_names 必须是字符串、列表或 pd.Series")

        job_names = list(map(clean_job_name, job_names))
        total_jobs = len(job_names)
        batches = [job_names[i: i + batch_size]
                   for i in range(0, total_jobs, batch_size)]
        results = []

        with tqdm(total=len(batches), desc="Processing Batches", unit="batch", disable=not show_progress) as pbar:
            for batch in batches:
                job_embeddings = self._encode_texts(batch)
                batch_results = self._match(job_embeddings, top_n, match_prt_level)
                results.extend([{"input": input, **match}
                               for input, match in zip(batch, batch_results)])
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
