"""增加整数格式的职业代码"""

import pandas as pd

csco_file = 'data/csco22.csv'
alias_file = 'data/csco22_aliases.csv'

csco_df = pd.read_csv(csco_file)
alias_df = pd.read_csv(alias_file)

csco_df['code_num'] = csco_df['code'].str.replace('-', '')
alias_df['csco_code_num'] = alias_df['csco_code'].str.replace('-', '')

csco_df.rename(columns={'title': 'name'}, inplace=True)

csco_df = csco_df[['level', 'code', 'code_num', 'name', 'definition', 'tasks']]
alias_df = alias_df[['alias', 'csco_code', 'csco_code_num', 'csco_name']]

csco_df.sort_values(by='code_num')
alias_df.sort_values(by='csco_code_num')

csco_df.to_csv(csco_file, index=False)
alias_df.to_csv(alias_file, index=False)