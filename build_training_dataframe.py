# Build dataframe for training and validation
import pandas as pd
import numpy as np
from itertools import product
sales = pd.read_csv('./data-readonly/sales_train.csv.gz')
# test = pd.read_csv('./data-readonly/test.csv.gz')
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int16'))

# turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int16)

# get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

# fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
# join aggregated data to the grid
df_All_train = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
# sort the data
df_All_train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
df_All_train.shop_id = df_All_train.shop_id.astype('int8')
df_All_train.date_block_num = df_All_train.date_block_num.astype('int8')
df_All_train.target = df_All_train.target.astype('int8')
df_All_train = df_All_train.reset_index(drop=True)
# df_All_train = df_All_train.to_sparse()
df_All_train.to_hdf('./HDF/All_train.hdf',key='train')

#%% generate validation
sales = pd.read_csv('./data-readonly/sales_train.csv.gz')
test = pd.read_csv('./data-readonly/test.csv.gz')
cur_shops = test['shop_id'].unique()
cur_items = test['item_id'].unique()
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from test
grid = []
for block_num in [28, 29, 30, 31, 32, 33]:  # use 6 months for validation
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int16'))

# turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int16)

# get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

# fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
# join aggregated data to the grid
df_Val = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
# sort the data
df_Val.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
df_Val.shop_id = df_Val.shop_id.astype('int8')
df_Val.date_block_num = df_Val.date_block_num.astype('int8')
df_Val.target = df_Val.target.astype('int8')
df_Val = df_Val.reset_index(drop=True)
# df_Val = df_Val.to_sparse()
df_Val.to_hdf('./HDF/All_train.hdf',key='val')