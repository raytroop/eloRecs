import pandas as pd

df1 = pd.read_csv('submission.csv')
df2 = pd.read_csv('submission0209-170347.csv')
df3 = pd.read_csv('submission_kernel.csv')
df4 = pd.read_csv('subm_3.651336_XGB_cv11_2019-02-07-16-05.csv')
df5 = pd.read_csv('subm_3.648358_LGBM_cv11_2019-02-08-05-49.csv')

df1['target'] = .2*df1['target'] + .2*df2['target'] + .2*df3['target'] + .2*df4['target'] + .2*df5['target']
df1.to_csv('ens.csv',index=False)