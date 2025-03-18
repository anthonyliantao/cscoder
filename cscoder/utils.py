import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def load_csco(version="csco22"):
    """加载 CSCO 数据"""
    file_path = os.path.join(DATA_DIR, f"{version}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 {file_path}，请确认 CSV 文件是否存在")

    df = pd.read_csv(file_path, usecols=["code_num", "name", "alias"], dtype=str)
    
    df.rename(columns={'code_num': 'code'}, inplace=True)
    df['alias'] = df["alias"].apply(lambda x: x.split(",") if isinstance(x, str) else [df["name"]])
    
    return df


if __name__ == "__main__":
    df = load_csco()
    print(df.head())
