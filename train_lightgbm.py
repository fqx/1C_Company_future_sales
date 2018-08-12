#%% a simple training script for lightGBM
import pandas as pd
import lightgbm as lgb

training_name = 'lightgbm'
X_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_x')
y_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_y')
X_test = pd.read_hdf('./HDF/Train_with_features.hdf', key='test_x')

lgb_params = {
               'feature_fraction': 0.7907245563968146,
               'metric': 'rmse',
               'nthread':1,
               'max_bin': 205,
               'min_data_in_leaf': 34,
               'min_sum_hessian_in_leaf': 2.1025393167025026,
               'bagging_fraction': 0.8462854127561101,
               'learning_rate': 0.00724657572276387,
               'objective': 'mse',
               'bagging_seed': 2**7,
               'num_leaves': 158,
               'bagging_freq': 3,
               'lambda_l1': 6.929773653631716,
               'lambda_l2': 7.47678994694134,
               'verbose':0
              }

model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)
pred_lgb = model.predict(X_test)

#%% make pred files
pred = pd.read_csv('./data-readonly/sample_submission.csv.gz')
pred['item_cnt_month'] = pred_lgb
pred['item_cnt_month'] = pred['item_cnt_month'].clip(lower=0, upper=20)
pred.to_csv('./Prediction/'+training_name+'.csv', index=False)