import pandas as pd
import numpy as np
from itertools import product

class FeatureGenerator():

    def __init__(self):
        self=self

    def create_promotion_bin(self, df_item_per_biweek,p1,p2,p3):
        #tuning
        def promotion_help(promotion,simulationPrice,ts1,ts2,ts3):
          if (promotion==0):
            return 0
          elif (simulationPrice<=ts1):
            return 1
          elif (simulationPrice<=ts2):
            return 2
          elif (simulationPrice<=ts3):
            return 3
          else:
            return 4

        ts1 = np.percentile(df_item_per_biweek['simulationPrice'].values, p1)
        ts2 = np.percentile(df_item_per_biweek['simulationPrice'].values, p2)
        ts3 = np.percentile(df_item_per_biweek['simulationPrice'].values, p3)
        df_item_per_biweek['promotion_bin'] = df_item_per_biweek.apply(lambda x: promotion_help(x['is_promotion'],x['simulationPrice'],ts1,ts2,ts3), axis=1)
        return df_item_per_biweek


    def create_features_to_lag(self, df_item_per_biweek, features, lags):
      for l in lags:
        df = df_item_per_biweek.copy()
        df.biweek = df.biweek + l
        df = df[['biweek','itemID']+features]
        df.columns = ['biweek','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
        df_item_per_biweek = pd.merge(df_item_per_biweek, df,on=['biweek','itemID'],how='left')
      return df_item_per_biweek

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
