#%% train
import pandas as pd

import KNN
KNN_feats = KNN.train(df_Train.drop(columns=['target']).values, df_Train['target'].clip(lower=0,upper=20).values)
columns = []
for i in range(KNN_feats.shape[1]):
    columns.append('KNN' + str(i))
KNN_feats = pd.DataFrame(KNN_feats, columns=columns)
KNN_feats.to_hdf('KNN_features.hdf', key='train')

#%% test set