import pandas as pd
import numpy as np
from itertools import product

class FeatureGenerator():

    def __init__(self):
        self=self

    def create_percentile_order_count(self, df_item_per_biweek, train_biweek):
        df = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek]
        df = df.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
        ts25 = np.percentile(df['order'].values, 25)
        ts50 = np.percentile(df['order'].values, 50)
        ts75 = np.percentile(df['order'].values, 75)
        df['percentile_order_count'] = df['order'].apply(lambda x: 1 if (x <= ts25) else 2 if ((x>ts25) & (x<=ts50)) else  3 if ((x>ts50) & (x<=ts75))  else 4)
        return pd.merge(df_item_per_biweek, df[['itemID', 'percentile_order_count']], on='itemID')

    def one_hot_encoding(self, df_item_per_biweek, column):

        return pd.get_dummies(df_item_per_biweek, columns=column)

    def create_features_to_lag(self, df_item_per_biweek, features, lags):
      for l in lags:
        df = df_item_per_biweek.copy()
        df.biweek = df.biweek + l
        df = df[['biweek','itemID']+features]
        df.columns = ['biweek','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
        df_item_per_biweek = pd.merge(df_item_per_biweek, df,on=['biweek','itemID'],how='left')
      return df_item_per_biweek

    def feature_agg_order_per_time(self, df_item_per_biweek, features, train_biweek):
      for feature in features:
        for col, agg, aggtype in [('order',np.sum,'sum'),('order',np.mean,'mean'),('order',np.median,'median')]:
          df2 = df_item_per_biweek[(df_item_per_biweek['cl_order']==1) & (df_item_per_biweek['biweek']<train_biweek)][[feature, 'biweek', col]].groupby([feature,'biweek']).aggregate(agg).reset_index()
          df2.columns = [feature, 'biweek', feature+'_'+aggtype+'_'+col+'_per_'+'biweek']
          df_item_per_biweek = pd.merge(df_item_per_biweek, df2, on=['biweek',feature], how='left')
      return df_item_per_biweek

    def create_max(self, df_item_per_biweek,train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek]
      column = []
      for i in range(1,10464):
           column.append(df['order'][(df['itemID']==i)].max())

      df_item_per_biweek['max_order'] = np.repeat(column,13)
      return df_item_per_biweek

    def create_median_orders_base(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 0)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
      df.columns = ['itemID','median_order_base']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_mean_orders_base(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 0)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
      df.columns = ['itemID','mean_order_base']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_median_orders_promotion(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 1)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
      df.columns = ['itemID','median_order_promotion']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_mean_orders_promotion(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 1)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
      df.columns = ['itemID','mean_order_promotion']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')
