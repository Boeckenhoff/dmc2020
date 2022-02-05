import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")


def load_train_test_dates(path):
  test_01_dates = pd.read_csv(path+'train_test_splits/01/test_01_dates.csv')
  test_02_dates = pd.read_csv(path+'train_test_splits/02/test_02_dates.csv')
  test_03_dates = pd.read_csv(path+'train_test_splits/03/test_03_dates.csv')
  test_04_dates = pd.read_csv(path+'train_test_splits/04/test_04_dates.csv')
  test_05_dates = pd.read_csv(path+'train_test_splits/05/test_05_dates.csv')
  test_06_dates = pd.read_csv(path+'train_test_splits/06/test_06_dates.csv')
  test_07_dates = pd.read_csv(path+'train_test_splits/07/test_07_dates.csv')
  test_08_dates = pd.read_csv(path+'train_test_splits/08/test_08_dates.csv')
  test_09_dates = pd.read_csv(path+'train_test_splits/09/test_09_dates.csv')
  test_10_dates = pd.read_csv(path+'train_test_splits/10/test_10_dates.csv')
  test_11_dates = pd.read_csv(path+'train_test_splits/11/test_11_dates.csv')

  train_01_dates = pd.read_csv(path+'train_test_splits/01/train_01_dates.csv')
  train_02_dates = pd.read_csv(path+'train_test_splits/02/train_02_dates.csv')
  train_03_dates = pd.read_csv(path+'train_test_splits/03/train_03_dates.csv')
  train_04_dates = pd.read_csv(path+'train_test_splits/04/train_04_dates.csv')
  train_05_dates = pd.read_csv(path+'train_test_splits/05/train_05_dates.csv')
  train_06_dates = pd.read_csv(path+'train_test_splits/06/train_06_dates.csv')
  train_07_dates = pd.read_csv(path+'train_test_splits/07/train_07_dates.csv')
  train_08_dates = pd.read_csv(path+'train_test_splits/08/train_08_dates.csv')
  train_09_dates = pd.read_csv(path+'train_test_splits/09/train_09_dates.csv')
  train_10_dates = pd.read_csv(path+'train_test_splits/10/train_10_dates.csv')
  train_11_dates = pd.read_csv(path+'train_test_splits/11/train_11_dates.csv')

  test_dates = [test_01_dates,test_02_dates,test_03_dates,test_04_dates,test_05_dates,test_06_dates,test_07_dates,test_08_dates,test_09_dates,test_10_dates,test_11_dates]
  train_dates = [train_01_dates, train_02_dates,train_03_dates,train_04_dates,train_05_dates,train_06_dates,train_07_dates,train_08_dates,train_09_dates,train_10_dates,train_11_dates]

  return train_dates, test_dates

def create_item_perDay(orders_df, infos_df, items_df):

  df = pd.DataFrame(list(product(infos_df['itemID'], pd.date_range(start='01-01-2018', end='06-29-2018'))), columns=['itemID', 'date'])

  date_df = pd.DataFrame(pd.date_range(start='01-01-2018', end='06-29-2018'), columns=['date'])
  date_df['week'] = date_df['date'].dt.week.shift(-2, fill_value=26)

  df = df.merge(date_df, how='outer', on='date')

  orders_df['date'] = pd.to_datetime(orders_df['time']).dt.date
  orders_df = orders_df.groupby(['date', 'itemID']).aggregate({'order':np.sum,'salesPrice':np.mean}).reset_index()
  orders_df['date'] = pd.to_datetime(orders_df['date'])

  df = pd.merge(df, orders_df, how='outer', on=['date','itemID'])

  df = df.drop(['salesPrice'], axis=1)
  df['day'] = pd.Index(df['date']).day
  df['dayofweek'] = pd.Index(df['date']).dayofweek
  df['month'] = pd.Index(df['date']).month
  df['dateInt'] = df['date'].astype(np.int64)

  df['order'].fillna(0, inplace=True)

  return df.merge(items_df, on='itemID')

def train_test_split(df_per_day,train_dates,test_dates):
  X_train = df_per_day[df_per_day['date'].isin(train_dates['date'].values)]
  y_train = X_train['order'].values
  X_train = X_train.drop(['order'], axis=1)
  X_test = df_per_day[df_per_day['date'].isin(test_dates['date'].values)]
  y_test = X_test['order'].values
  X_test = X_test.drop(['order'], axis=1)

  return X_train, X_test, y_train, y_test

def create_item_perWeek(orders_df, infos_df, items_df):

  df = pd.DataFrame(list(product(infos_df['itemID'], pd.date_range(start='01-01-2018', end='06-29-2018'))), columns=['itemID', 'date'])

  date_df = pd.DataFrame(pd.date_range(start='01-01-2018', end='06-29-2018'), columns=['date'])
  date_df['week'] = date_df['date'].dt.week.shift(-2, fill_value=26)

  df = df.merge(date_df, how='outer', on='date')

  orders_df['date'] = pd.to_datetime(orders_df['time']).dt.date
  orders_df = orders_df.groupby(['date', 'itemID']).aggregate({'order':np.sum,'salesPrice':np.mean}).reset_index()
  orders_df['date'] = pd.to_datetime(orders_df['date'])

  df = pd.merge(df, orders_df, how='outer', on=['date','itemID'])

  df = df.groupby(['week', 'itemID']).aggregate({'order':np.sum,'salesPrice':np.mean}).reset_index()
  df = df.drop(['salesPrice'], axis=1)

  df['order'].fillna(0, inplace=True)

  return df.merge(items_df, on='itemID')
