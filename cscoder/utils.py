import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def load_csco(version="csco22"):
    """加载 CSCO 数据"""
    file_path = os.path.join(DATA_DIR, f"{version}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 {file_path}，请确认 CSV 文件是否存在")

    return pd.read_csv(file_path, usecols=["code_num", "name"], dtype={"code_num": str})


def load_aliases(version="csco22"):
    """加载 CSCO 别名数据"""
    file_path = os.path.join(DATA_DIR, f"{version}_aliases.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 {file_path}，请确认 CSV 文件是否存在")

    df = pd.read_csv(file_path).dropna()
    df["alias"] = df["alias"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["alias"])

    return df
