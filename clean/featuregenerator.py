import pandas as pd
import numpy as np
from itertools import product

class FeatureGenerator():

    def __init__(self):
        self=self

    def create_percentile_order_count(self, df_item_per_biweek, train_biweek):
        #tuning
        df = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek]
        df = df.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
        ts25 = np.percentile(df['order'].values, 25)
        ts50 = np.percentile(df['order'].values, 50)
        ts75 = np.percentile(df['order'].values, 75)
        df['percentile_order_count'] = df['order'].apply(lambda x: 0 if (x==0) else 1 if ((x>0) & (x <= ts25)) else 2 if ((x>ts25) & (x<=ts50)) else  3 if ((x>ts50) & (x<=ts75))  else 4)
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
        for col, agg, aggtype in [('order',np.sum,'sum'),('order',np.mean,'mean'),('order',np.median,'median'),('order',np.max,'max')]:
          df2 = df_item_per_biweek[(df_item_per_biweek['order']>0) & (df_item_per_biweek['biweek']<train_biweek)][[feature, 'biweek', col]].groupby([feature,'biweek']).aggregate(agg).reset_index()
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
      df = df[(df['is_promotion'] == 0) & (df['order']>0)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
      df.columns = ['itemID','median_order_base']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_mean_orders_base(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 0)& (df['order']>0)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
      df.columns = ['itemID','mean_order_base']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_median_orders_promotion(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 1)& (df['order']>0)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
      df.columns = ['itemID','median_order_promotion']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_mean_orders_promotion(self, df_item_per_biweek, train_biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
      df = df[(df['is_promotion'] == 1)& (df['order']>0)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
      df.columns = ['itemID','mean_order_promotion']
      return df_item_per_biweek.merge(df,on = ['itemID'],how='outer')

    def create_cl_order_mean(self, df_item_per_biweek,biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek']<biweek]
      df = df.groupby(['cl_order']).aggregate({'order':np.mean}).reset_index()
      df.columns = ['cl_order','cl_order_mean']
      return df_item_per_biweek.merge(df,on= 'cl_order', how = 'left')

    def create_cl_order_median(self, df_item_per_biweek,biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek']<biweek]
      df = df.groupby(['cl_order']).aggregate({'order':np.median}).reset_index()
      df.columns = ['cl_order','cl_order_median']
      return df_item_per_biweek.merge(df,on= 'cl_order', how = 'left')

    def create_cl_order_max(self, df_item_per_biweek,biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek']<biweek]
      df = df.groupby(['cl_order']).aggregate({'order':'max'}).reset_index()
      df.columns = ['cl_order','cl_order_max']
      return df_item_per_biweek.merge(df,on= 'cl_order', how = 'left')

    def create_cl_order_min(self, df_item_per_biweek,biweek):
      df = df_item_per_biweek[df_item_per_biweek['biweek']<biweek]
      df = df.groupby(['cl_order']).aggregate({'order':'min'}).reset_index()
      df.columns = ['cl_order','cl_order_min']
      return df_item_per_biweek.merge(df,on= 'cl_order', how = 'left')

    def create_first_time_in_promotion(self, df_item_per_biweek):
      df = df_item_per_biweek[['itemID','biweek']][df_item_per_biweek['is_promotion']>0]
      df = df.groupby('itemID').min()
      df['firstTimeInPromotion']=1
      df_item_per_biweek = pd.merge(df_item_per_biweek,df,on=['itemID','biweek'], how='left')
      df_item_per_biweek['firstTimeInPromotion'].fillna(0, inplace=True)
      return df_item_per_biweek

    def create_first_time_sold(self, df_item_per_biweek, train_biweek):
        df = df_item_per_biweek[['itemID','biweek']][(df_item_per_biweek['biweek'] < train_biweek) & (df_item_per_biweek['order'] > 0)]
        df = df.groupby('itemID').min()
        df['firstTimeSold'] = 1
        df_item_per_biweek = pd.merge(df_item_per_biweek, df, on=['itemID', 'biweek'], how='left')
        df_item_per_biweek['firstTimeSold'].fillna(0, inplace=True)
        return df_item_per_biweek

    def create_mean_median_orders_promotion_bins(self, df_item_per_biweek, train_biweek):
        df = df_item_per_biweek[df_item_per_biweek['biweek'] < train_biweek]
        df = df[(df['is_promotion'] == 1)& (df['order']>0)].groupby(['promotion_bin']).aggregate({'order':[np.median,np.mean,'max']}).reset_index()
        df.columns = ['promotion_bin','mean_order_promotion_bin', 'median_order_promotion_bin', 'max_order_promotion_bin']
        return df_item_per_biweek.merge(df,on = ['promotion_bin'],how='left')

    def create_similarItemsInPromotion_per_biweek(self, df_item_per_biweek,similiar_items_in_promotion_df):
      # Muessen wir die test wochen rausnehmen?
      # wie aggregieren wir similarProductsInPromotionProportion?
      date_df = pd.DataFrame(pd.date_range(start='01-01-2018', end='06-29-2018'), columns=['date'])
      date_df['week'] = date_df['date'].dt.week.shift(-2, fill_value=26)
      similiar_items_in_promotion_df['date'] = pd.to_datetime(similiar_items_in_promotion_df['date'])
      similiar_items_in_promotion_df = similiar_items_in_promotion_df.merge(date_df, how='outer', on='date')


      similiar_items_in_promotion_df = similiar_items_in_promotion_df.groupby(['itemID', 'week']).aggregate({'similarProductIsInPromotion':np.max,'similarProductsInPromotionCount':np.sum,'similarProductsInPromotionProportion':np.sum}).reset_index()

      similiar_items_in_promotion_df['biweek'] = np.tile(np.repeat(range(1,14),2),10463)
      similiar_items_in_promotion_df = similiar_items_in_promotion_df.groupby(['itemID', 'biweek']).aggregate({'similarProductIsInPromotion':np.max,'similarProductsInPromotionCount':np.sum,'similarProductsInPromotionProportion':np.sum}).reset_index()


      return df_item_per_biweek.merge(similiar_items_in_promotion_df, on = ['itemID','biweek'])
