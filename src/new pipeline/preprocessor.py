import pandas as pd
import numpy as np
from itertools import product
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class PreProcessor():

    def __init__(self):
        self=self

    def create_item_per_biweek(self,orders_df, items_df, infos_df,promo,siip_df,shift):
        #one row per itemID and date
        df = pd.DataFrame(list(product(infos_df['itemID'], pd.date_range(start='01-01-2018', end='06-29-2018'))), columns=['itemID', 'date'])

        #switch week to dmc-week (Mo-So to Sa-Fr)
        date_df = pd.DataFrame(pd.date_range(start='01-01-2018', end='06-29-2018'), columns=['date'])
        date_df['week'] = date_df['date'].dt.week.shift(-2, fill_value=26)
        df = df.merge(date_df, how='outer', on='date')

        #group all transaction to date
        orders_df['date'] = pd.to_datetime(orders_df['time']).dt.date
        orders_df = orders_df.groupby(['date', 'itemID']).aggregate({'order':np.sum,'salesPrice':np.mean}).reset_index()
        orders_df['date'] = pd.to_datetime(orders_df['date'])
        df = pd.merge(df, orders_df, how='outer', on=['date','itemID'])

        df.sort_values(by=['itemID','date'],inplace=True)

        #add promotion
        df['is_promotion'] = promo

        #add similarProductIsInPromotion
        siip_df['date'] = pd.to_datetime(siip_df['date'])
        df = pd.merge(df, siip_df[['itemID','date','similarProductsInPromotionCount']], how='outer', on=['date','itemID'])

        #group all transaction by week
        df = df.groupby(['itemID','week']).aggregate({'order':np.sum,'salesPrice':np.mean,'is_promotion':np.sum,'similarProductsInPromotionCount':np.sum}).reset_index()
        #create biweek
        if (shift):
          df = df[df['week']<26]
          df['biweek'] = np.tile([1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],10463)
        else:
          df['biweek'] = np.tile(np.repeat(range(1,14),2),10463)

        #group all transaction by biweek
        df = df.groupby(['itemID', 'biweek']).aggregate({'order':np.sum,'salesPrice':np.mean,'is_promotion':np.sum,'similarProductsInPromotionCount':np.sum}).reset_index()

        df = df.drop(['salesPrice'], axis=1)
        df['order'].fillna(0, inplace=True)
        df['similarProductsInPromotionCount'].fillna(0, inplace=True)
        df['is_promotion'].fillna(0, inplace=True)
        df.dropna(inplace=True)
        df = pd.merge(df, infos_df[['itemID','simulationPrice']], on='itemID')

        return df.merge(items_df, on='itemID')

    def create_classification(self, df_item_per_biweek):
        #create feature for classification
        df_item_per_biweek['cl_order'] = df_item_per_biweek['order'].apply(lambda x: 1 if x > 0 else 0)
        return df_item_per_biweek

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

    def create_classification_bins(self, df_item_per_biweek,biweek):
        #tuning
            df = df_item_per_biweek[df_item_per_biweek['biweek']<biweek]
            df = df[df['order']>0]
            ts1 = np.percentile(df['order'].values, 25)
            ts2 = np.percentile(df['order'].values, 50)
            ts3 = np.percentile(df['order'].values, 75)

            #create feature for classification_bins

            df_item_per_biweek['cl_order'] = df_item_per_biweek['order'].apply(lambda x: 0 if (x == 0) else 1 if ((x>0) & (x<=ts1)) else 2 if ((x>ts1) & (x<=ts2)) else  3 if ((x>ts2) & (x<=ts3))  else 4)

            return df_item_per_biweek

    def create_promotion(self, df_item_per_biweek):
        #create featuer if itemID+order is peak
      df = df_item_per_biweek
      column = []
      for i in range(1,10464):
        ts = np.percentile(df['order'][df['itemID']==i],90)
        column = column + list(df['order'][df['itemID']==i].apply(lambda x: 1 if (x > ts) else 0))

      df_item_per_biweek['is_promotion'] = column
      return df_item_per_biweek

    def train_test_split_cl(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order','cl_order'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order','cl_order'], axis=1)
        y_train = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']==train_biweek].values
        return X_train, X_test, y_train, y_test

    def train_test_split_cl_rdm_undersample(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order','cl_order'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order','cl_order'], axis=1)
        y_train = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']==train_biweek].values
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=0, replacement=False)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def train_test_split_cl_smote_oversample(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order','cl_order'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order','cl_order'], axis=1)
        y_train = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']==train_biweek].values
        X_train = X_train.fillna(0)
        y_train = pd.DataFrame(y_train)
        y_train = y_train.fillna(0)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def train_test_split_cl_promotion_smote_oversample(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order','is_promotion'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order','is_promotion'], axis=1)
        y_train = df_item_per_biweek['is_promotion'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['is_promotion'][df_item_per_biweek['biweek']==train_biweek].values
        X_train = X_train.fillna(0)
        y_train = pd.DataFrame(y_train)
        y_train = y_train.fillna(0)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def train_test_split_ordersize_cl(self, df_item_per_biweek, train_biweek):
        for i in range(1, 14):
            df_item_per_currentWeek= df_item_per_biweek.loc[(df_item_per_biweek['order'] > 0) & (df_item_per_biweek['biweek'] == i)]
            ts25 = np.percentile(df_item_per_currentWeek['order'].values, 25)
            ts50 = np.percentile(df_item_per_currentWeek['order'].values, 50)
            ts75 = np.percentile(df_item_per_currentWeek['order'].values, 75)
            df_item_per_currentWeek= df_item_per_biweek.loc[df_item_per_biweek['biweek'] == i]
            #0: kein mal 1: erstes Quantil 2: zweites Quantil 3: dritte Quantil 4: viertes Quantil
            df_item_per_currentWeek['clordersize'] = df_item_per_currentWeek['order'].apply(lambda x: 0 if (x == 0) else 1 if (x <= ts25) else 2 if ((x>ts25) & (x<=ts50)) else  3 if ((x>ts50) & (x<=ts75))  else 4)
            if i == 1:
                a = df_item_per_currentWeek[['itemID', 'biweek', 'clordersize']]
            else:
                frames = [a,df_item_per_currentWeek[['itemID', 'biweek', 'clordersize']] ]
                a = pd.concat(frames,ignore_index=True)
        df_item_per_biweek = df_item_per_biweek.merge(a, on=['itemID', 'biweek'])
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order','cl_order','percentile_order_count','clordersize'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order','cl_order','percentile_order_count','clordersize'], axis=1)
        y_train = df_item_per_biweek['clordersize'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['clordersize'][df_item_per_biweek['biweek']==train_biweek].values
        return X_train, X_test, y_train, y_test


    def train_test_split(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order'], axis=1)
        y_train = df_item_per_biweek['order'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['order'][df_item_per_biweek['biweek']==train_biweek].values
        return X_train, X_test, y_train, y_test
