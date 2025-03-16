"""临时python脚本 用于修复现在的职业别名库不能覆盖所有职业的问题"""

import pandas as pd

csco_file = 'data/csco22.csv'
alias_file = 'data/csco22_aliases.csv'

csco_df = pd.read_csv(csco_file)
alias_df = pd.read_csv(alias_file)

csco_df['alias'] = csco_df['name']
csco_df.rename(columns={'code': 'csco_code', 'code_num': 'csco_code_num', 'name': 'csco_name'}, inplace=True)
complete_df = csco_df[['alias', 'csco_code', 'csco_code_num', 'csco_name']]

alias_df = pd.concat([alias_df, complete_df])
alias_df = alias_df.drop_duplicates()
alias_df = alias_df.sort_values(by='csco_code_num')

alias_df.to_csv(alias_file, index=False)
