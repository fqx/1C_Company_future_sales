#%% train
import pandas as pd
if __name__ == '__main__':
    df_Train = pd.read_hdf('./HDF/All_train.hdf', key='train')

    import KNN_tf as KNN
    KNN_feats = KNN.train(df_Train.drop(columns=['target']).values, df_Train['target'].clip(lower=0,upper=20).values,
                          n_jobs=1)
    columns = []
    for i in range(KNN_feats.shape[1]):
        columns.append('KNN' + str(i))
    KNN_feats = pd.DataFrame(KNN_feats, columns=columns)
    KNN_feats.to_hdf('./HDF/KNN_features.hdf', key='train')

#%% test set
