from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import load_csco
from .preprocess import clean_job_name


class CSCOder:
    """
    匹配最接近的中国标准职业分类（CSCO）代码。

    该类提供了加载模型、处理职业名称、计算相似度以及返回匹配结果的功能。
    """

    def __init__(self, version="csco22", model_name="paraphrase-multilingual-MiniLM-L12-v2", disable_cache=False):
        """
        初始化 CSCOder 实例。

        :param version: CSCO 数据的版本，默认为 "csco22"。
        :param model_name: 用于文本嵌入的模型名称，默认为 "paraphrase-multilingual-MiniLM-L12-v2"。
        :param disable_cache: 是否禁用缓存，默认开启。
        """
        self.model_name = model_name
        self.version = version
        self.disable_cache = disable_cache
        self._model = None
        self._csco_data = None
        self._alias_data = None
        self._cache = OrderedDict() if not disable_cache else None
        self.cache_size = 500000 if not disable_cache else None

    @property
    def model(self):
        """加载或返回 SentenceTransformer 模型实例。"""
        if self._model is None:
            print(f"Loading model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def csco_data(self):
        """加载或返回 CSCO 代码到名称的映射字典。"""
        if self._csco_data is None:
            df = load_csco(self.version)
            self._csco_data = df.set_index("code")["name"].to_dict()
        return self._csco_data

    @property
    def alias_data(self):
        """加载或返回别名数据及其嵌入向量。"""
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
        """返回职业别名数据的 DataFrame。"""
        return self.alias_data[0]

    @property
    def alias_embeddings(self):
        """返回职业别名数据的嵌入矩阵。"""
        return self.alias_data[1]

    def _store_in_cache(self, texts, vectors):
        """存入缓存 自动淘汰最久未使用的项"""
        for text, vector in zip(texts, vectors):
            self._cache[text] = vector
            self._cache.move_to_end(text)

            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)

    def _encode_texts(self, texts, *args, **kwargs):
        """将文本编码为向量。"""
        if self.disable_cache:
            return np.array(self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, *args, **kwargs))

        results = [None] * len(texts)
        text_positions = {}  # 用于记录未缓存文本的索引位置
        texts_to_encode = list(
            {text for text in texts if text not in self._cache})

        # 遍历 texts，查找缓存命中，并收集未缓存文本
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                text_positions.setdefault(text, []).append(i)

        # 计算未缓存文本向量
        if texts_to_encode:
            vectors = self.model.encode(
                texts_to_encode, convert_to_numpy=True, normalize_embeddings=True, *args, **kwargs
            )
            self._store_in_cache(texts_to_encode, vectors)

            for text, vector in zip(texts_to_encode, vectors):
                for i in text_positions[text]:
                    results[i] = vector

        return np.array(results)

    def _match(self, job_embeddings, top_n=1, match_prt_lvl=False):
        """
        计算相似度并匹配。

        :param job_embeddings: 职业名称的嵌入向量。
        :param top_n: 返回最相似的前 N 个匹配，默认为 1。
        :param match_prt_lvl: 是否根据相似度调整匹配层级，默认为 False。
        :return: 匹配结果的列表。
        """
        similarity_matrix = 1 - \
            cdist(job_embeddings, self.alias_embeddings, metric="cosine")

        results = []
        for scores in similarity_matrix:
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:top_n]
            for idx in top_indices:
                sim_score = scores[idx]
                csco_code = self.alias_df.iloc[idx]["code"]
                csco_code = self._match_parent_level(
                    csco_code, sim_score) if match_prt_lvl else csco_code
                csco_name = self.csco_data.get(str(csco_code))
                results.append(
                    {
                        "matched_code": csco_code,
                        "matched_name": csco_name,
                        "similarity": sim_score
                    })

        return results

    def _match_parent_level(self, csco_code, sim_score):
        """
        根据相似度返回对应层级的父级代码。

        :param csco_code: 匹配的 CSCO 代码。
        :param sim_score: 相似度分数。
        :return: 调整后的 CSCO 代码。
        """
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
        """
        匹配单个职业并返回匹配结果。

        :param job_name: 职业名称。
        :param top_n: 返回最相似的前 N 个匹配，默认为 1。
        :param return_df: 是否返回 DataFrame 格式，默认为 True。
        :param match_prt_level: 是否根据相似度调整匹配层级，默认为 False。
        :return: 匹配结果，列表或 DataFrame。
        """
        if not job_name.strip():
            return []

        job_embedding = self._encode_texts([clean_job_name(job_name)])
        match_results = self._match(job_embedding, top_n, match_prt_level)
        results = [{"input": job_name, **match} for match in match_results]

        return results if not return_df else pd.DataFrame(results)

    def find_best_matches(self, job_names, top_n=1, batch_size=1000, return_df=True, show_progress=False, match_prt_level=False):
        """
        批量匹配多个职业并返回匹配结果。

        :param job_names: 职业名称列表或字符串。
        :param top_n: 返回最相似的前 N 个匹配，默认为 1。
        :param batch_size: 每批处理的职业数量，默认为 1000。
        :param return_df: 是否返回 DataFrame 格式，默认为 True。
        :param show_progress: 是否显示进度条，默认为 False。
        :param match_prt_level: 是否根据相似度调整匹配层级，默认为 False。
        :return: 匹配结果，列表或 DataFrame。
        """
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
                batch_results = self._match(
                    job_embeddings, top_n, match_prt_level)
                results.extend([{"input": input, **match}
                               for input, match in zip(batch, batch_results)])
                pbar.update(1)

        return pd.DataFrame(results) if return_df else results


if __name__ == "__main__":
    coder = CSCOder()

    # 匹配单个职业示例
    result = coder.find_best_match("软件工程师")
    print(result)

    # 匹配多个职业示例
    job_list = ["数据分析师", "产品经理", "注册会计师", "Java工程师"]
    result = coder.find_best_matches(job_list)
    print(result)
