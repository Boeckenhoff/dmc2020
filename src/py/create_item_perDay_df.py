import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm_notebook

def create_item_perDay_df(orders_df, infos_df, items_df):

  df = pd.DataFrame(list(product(infos_df['itemID'], pd.date_range(start='01-01-2018', end='06-29-2018'))), columns=['itemID', 'date'])

  date_df = pd.DataFrame(pd.date_range(start='01-01-2018', end='06-29-2018'), columns=['date'])
  date_df['week'] = date_df['date'].dt.week.shift(-2, fill_value=26)

  df = df.merge(date_df, how='outer', on='date')

  orders_df['date'] = pd.to_datetime(orders_df['time']).dt.date
  orders_df = orders_df.groupby(['date', 'itemID']).aggregate({'order':np.sum,'salesPrice':np.mean}).reset_index()
  orders_df['date'] = pd.to_datetime(orders_df['date'])

  df = pd.merge(df, orders_df, how='outer', on=['date','itemID'])

  for i in tqdm_notebook(range(1,10464,1)):
    item = df.loc[df['itemID'] == i]
    item['salesPrice'].fillna(method='ffill', inplace=True)
    item['salesPrice'].fillna(method='bfill', inplace=True)
    item.fillna(infos_df['simulationPrice'].loc[infos_df['itemID']==i], inplace=True)
    df.loc[df['itemID'] == i] = item


  df.fillna(0, inplace=True)

  return df.merge(items_df, how='outer', on='itemID')
