#%% tried the mean of three preditions
import pandas as pd
col = 'item_cnt_month'
training_name = 'mean_ensemble_2'
df1 = pd.read_csv('./Prediction/linear.csv')
df2 = pd.read_csv('./Prediction/mean_ensemble.csv')

df4 = (df1[col] + df2[col]) / 2
pred = pd.read_csv('./data-readonly/sample_submission.csv.gz')
pred['item_cnt_month'] = df4
pred['item_cnt_month'] = pred['item_cnt_month'].clip(lower=0, upper=20)
pred.to_csv('./Prediction/'+training_name+'.csv', index=False)