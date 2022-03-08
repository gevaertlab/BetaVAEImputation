import pandas as pd

df = pd.read_csv('../data/data_complete.csv')
df_miss = pd.read_csv('../data/LGGGBM_missing_10perc_trial1.csv')
print(f'df complete: {df.shape}, df_miss: {df_miss.shape}')
# df_miss has approximately 2% missing data in each column
# 20% of rows have at least one missing value
drop_rows = df_miss[df_miss.isnull().any(axis=1)].index
non_missing_are_same = df_miss.drop(drop_rows).equals(df.drop(drop_rows).iloc[:,4:]) # df has 4 additional columns at the beginning
print(f'the non missing rows are the same {non_missing_are_same}')
breakpoint_var=True