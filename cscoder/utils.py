import pandas as pd
import importlib.resources


def load_csco(version="csco22"):
    """加载 CSCO 数据"""
    file_path = importlib.resources.files("cscoder.data").joinpath(f"{version}.csv")
    
    if not file_path.is_file():
        raise FileNotFoundError(f"未找到 {file_path}，请确认 CSV 文件是否存在")
    
    with file_path.open("r", encoding="utf-8") as f:
        df = pd.read_csv(f, usecols=["code_num", "name", "alias"], dtype=str)
    
    df.rename(columns={'code_num': 'code'}, inplace=True)
    df['alias'] = df["alias"].apply(lambda x: x.split(",") if isinstance(x, str) else [df["name"]])
    
    return df


if __name__ == "__main__":
    df = load_csco()
    print(df.head())
