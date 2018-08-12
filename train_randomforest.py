import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

training_name = 'randomforest'
X_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_x')
y_train = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_y')
X_test = pd.read_hdf('./HDF/Train_with_features.hdf', key='test_x')

model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
pred_rf = model.predict(X_test)

plt.barh(X_train.columns, model.feature_importances_)
plt.show()

#%% make pred files
pred = pd.read_csv('./data-readonly/sample_submission.csv.gz')
pred['item_cnt_month'] = pred_rf
pred['item_cnt_month'] = pred['item_cnt_month'].clip(lower=0, upper=20)
pred.to_csv('./Prediction/'+training_name+'.csv', index=False)

from sklearn.externals import joblib
joblib.dump(model, './Model/RandomForest.pkl', compress=3)