import pandas as pd
import json

# 读取Excel文件，将"NA"解析为NaN
anno_df = pd.read_excel('/Users/liantao/Desktop/testcases.xlsx', na_values=['NA'])

# 重命名列
anno_df.rename(columns={'text1': 'job_name', 'text2': 'csco_name'}, inplace=True)

# 读取CSCO数据
csco_df = pd.read_csv('data/csco22.csv')
csco_df.rename(columns={'name': 'csco_name', 'code_num': 'csco_code'}, inplace=True)

# 合并数据
anno_df = anno_df.merge(csco_df[['csco_name', 'csco_code']], on='csco_name', how='left')

# 将csco_code中的NaN填充为0并转换为整数
anno_df['csco_code'] = anno_df['csco_code'].fillna(0).astype(int)

# 构建测试用例
testcases = []
for _, row in anno_df.iterrows():
    # 显式将NaN转换为None
    csco_name = None if pd.isna(row['csco_name']) else row['csco_name']
    testcases.append({
        "input": row['job_name'],
        "expected": [{"csco_code": row['csco_code'], "csco_name": csco_name}]
    })

# 保存为JSON文件
with open('tests/data/testcases.json', 'w') as f:
    json.dump(testcases, f, ensure_ascii=False, indent=2)