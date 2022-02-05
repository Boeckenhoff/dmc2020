import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from itertools import product
import warnings
warnings.filterwarnings("ignore")

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

  df.fillna(0, inplace=True)

  return df.merge(items_df, on='itemID')


def generate_itemID_order_dist_per_weekday(df_per_day, train_dates):

  df1 = df_per_day[df_per_day['date'].isin(train_dates['date'].values)]
  final = df1.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
  final = pd.merge(final, df1.groupby(['itemID','dayofweek']).aggregate({'order':np.sum}).unstack(), on='itemID', how='left')
  for i in range(1,8,1):
    final['%'+str(i)] = final.iloc[:,(i+1)]/final.iloc[:,1]
  final = final[['itemID', 'order','%1','%2', '%3', '%4', '%5', '%6', '%7']]
  final.fillna(0, inplace=True)
  final = final[final['order']>0]
  k_means = KMeans(n_clusters=7)
  model = k_means.fit(final.iloc[:,2:8])
  pred= k_means.predict(final.iloc[:,2:8])
  final['itemID_order_dist_per_weekday'] = pred
  df1 = pd.DataFrame(range(1,10464,1), columns=['itemID'])
  df1 = pd.merge(df1,final[['itemID','itemID_order_dist_per_weekday']], on='itemID', how='left')
  df1.fillna(7, inplace=True)

  return df1[['itemID','itemID_order_dist_per_weekday']]

def train_test_split(df_per_day,train_dates,test_dates):
  X_train = df_per_day[df_per_day['date'].isin(train_dates['date'].values)]
  y_train = X_train['order'].values
  X_train = X_train.drop(['order','date'], axis=1)
  X_test = df_per_day[df_per_day['date'].isin(test_dates['date'].values)]
  y_test = X_test['order'].values
  X_test = X_test.drop(['order','date'], axis=1)

  return X_train, X_test, y_train, y_test
