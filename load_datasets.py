import pandas as pd
HDF = './HDF/Train_with_features.hdf'


def train(max_date_block_num=33, min_date_block_num=0):

    X_train = pd.read_hdf(HDF, key='train_x')
    y_train = pd.read_hdf(HDF, key='train_y')
    y_train = y_train[(min_date_block_num <= X_train['date_block_num']) & (X_train['date_block_num'] <= max_date_block_num)]
    X_train = X_train[(min_date_block_num <= X_train['date_block_num']) & (X_train['date_block_num'] <= max_date_block_num)]
    return X_train, y_train


def val(max_date_block_num=33, min_date_block_num=28):

    X_val = pd.read_hdf(HDF, key='train_x')
    y_val = pd.read_hdf(HDF, key='train_y')
    y_val = y_val[(min_date_block_num <= X_val['date_block_num']) & (X_val['date_block_num'] <= max_date_block_num)]
    X_val = X_val[(min_date_block_num <= X_val['date_block_num']) & (X_val['date_block_num'] <= max_date_block_num)]
    return X_val, y_val


def test():
    X_test = pd.read_hdf(HDF, key='test_x')
    return X_test
