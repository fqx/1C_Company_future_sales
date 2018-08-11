import keras as K
import pandas as pd


def rmse(y_true, y_pred):
    return K.backend.sqrt(K.backend.mean(K.backend.square(y_pred - y_true), axis=-1))

batch_size =2048
training_name = 'DNN'
X_All = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_x')
y_All = pd.read_hdf('./HDF/Train_with_features.hdf', key='train_y')
X_test = pd.read_hdf('./HDF/Train_with_features.hdf', key='test_x')

X_train = X_All[X_All['date_block_num'] < 33]
y_train = y_All[X_All['date_block_num'] < 33]
X_val = X_All[X_All['date_block_num'] == 33]
y_val = y_All[X_All['date_block_num'] == 33]

K.backend.clear_session()
model = K.Sequential()
model.add(K.layers.Dense(400, activation='relu', input_dim=156))
model.add(K.layers.Dense(100,activation='relu'))
model.add(K.layers.Dense(20,activation='relu'))
model.add(K.layers.Dense(1))

callbacks = K.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])
model.fit(x=X_train.values, y=y_train.values, validation_data=(X_val.values, y_val.values),
          callbacks=[callbacks],batch_size=batch_size, epochs=10)
pred_NN = model.predict(x=X_test.values, batch_size=batch_size)

#%% make pred files
pred = pd.read_csv('./data-readonly/sample_submission.csv.gz')
pred['item_cnt_month'] = pred_NN
pred['item_cnt_month'] = pred['item_cnt_month'].clip(lower=0, upper=20)
pred.to_csv('./Prediction/'+training_name+'.csv', index=False)