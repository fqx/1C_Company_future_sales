#%% validation
import pandas as pd

df_Test = pd.read_csv('./data-readonly/test.csv.gz')
df_Train = pd.read_hdf('./HDF/All_train.hdf', key='train')
df_Test = df_Test.drop(columns='ID')
df_Test['date_block_num'] = 34

#%%lag features
index_cols = ['shop_id', 'item_id', 'date_block_num']
cols_to_rename = list(df_Train.columns.difference(index_cols))
shift_range = [1, 2, 3, 4, 6, 12]
for month_shift in shift_range:
    train_shift = df_Train[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    df_Test = pd.merge(df_Test, train_shift, on=index_cols, how='left').fillna(0)

#%% holidays
df_Test['num_holidays'] = 1

#%% mean encoding
df_Items = pd.read_csv('./data-readonly/items.csv')
global_mean = df_Train.target.mean()
df_Train = df_Train.merge(df_Items[['item_id','item_category_id']], on='item_id')
df_Test = df_Test.merge(df_Items[['item_id','item_category_id']], on='item_id')

for id in ['item_id', 'shop_id', 'item_category_id']:
    mean = df_Train.groupby(id)['target'].mean()

    df_Test['{}_target_enc'.format(id)]= mean
    df_Test['{}_target_enc'.format(id)].fillna(global_mean, inplace=True)

#%% merge features
df_Items = pd.read_hdf('./HDF/features.hdf', key='items')
df_Items_categories = pd.read_hdf('./HDF/features.hdf', key='item_categories')
df_Shops = pd.read_hdf('./HDF/features.hdf', key='shops')
df_Release = pd.read_hdf('./HDF/features.hdf', key='release_date')

merge_list = [
    (df_Items_categories, 'item_category_id'),
    (df_Items, 'item_id'),
    (df_Shops, 'shop_id'),
    (df_Release, 'item_id')
]
for df, id in merge_list:
    df_Test = df_Test.merge(df, on=id)

df_Test.to_hdf('./HDF/Train_with_features.hdf', key='test_x')