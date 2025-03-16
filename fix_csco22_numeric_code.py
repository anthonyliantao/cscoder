"""修复csco22只有字符型的编码的问题"""

import pandas as pd

csco_df = pd.read_csv('data/csco22.csv')
alias_df = pd.read_csv('data/csco22_aliases.csv')

csco_df['code_num'] = csco_df['code'].str.replace('-', '').astype(int)
alias_df['csco_code_num'] = alias_df['csco_code'].str.replace('-', '').astype(int)

csco_df = csco_df[['level', 'code', 'code_num','title', 'definition', 'tasks']]
alias_df = alias_df[['alias', 'csco_code', 'csco_code_num', 'csco_name']]

csco_df.sort_values(by='code_num', inplace=True)
alias_df.sort_values(by='csco_code_num', inplace=True)

csco_df.to_csv('data/csco22.csv', index=False)
alias_df.to_csv('data/csco22_aliases.csv', index=False)