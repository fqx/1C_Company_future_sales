import pandas as pd
from sklearn.linear_model import LinearRegression

training_name = 'linear'
X_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_x')
y_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_y')
X_test = pd.read_hdf('./HDF/Train_with_features.hdf', key='test_x')

model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)
pred_lin = model.predict(X_test)

#%% make pred files
pred = pd.read_csv('./data-readonly/sample_submission.csv.gz')
pred['item_cnt_month'] = pred_lin
pred['item_cnt_month'] = pred['item_cnt_month'].clip(lower=0, upper=20)
pred.to_csv('./Prediction/'+training_name+'.csv', index=False)
