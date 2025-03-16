import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def load_csco_aliases(version="csco22"):
    """加载 CSCO 职业别名数据"""
    file_path = os.path.join(DATA_DIR, f"{version}_aliases.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 {file_path}，请确认 CSV 文件是否存在")

    df = pd.read_csv(file_path)

    df = df.dropna()
    df["alias"] = df["alias"].str.strip().str.lower()

    return df


def get_csco_name(csco_code, version="csco22"):
    """根据 CSCO 代码查询职业名称"""
    df = load_csco_aliases(version)
    csco_code_7d = int(str(csco_code).replace('-', '').ljust(7, '0'))
    result = df[df["csco_code_num"] == csco_code_7d]["csco_name"].unique()
    return result[0] if len(result) > 0 else None


# 测试代码
if __name__ == "__main__":
    df = load_csco_aliases()
    print(df.head())  # 查看前几行数据
    print(get_csco_name("1-00-00-00"))  # 测试查询功能
