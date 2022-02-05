import pandas as pd
import numpy as np
from itertools import product

class PreProcessor():

    def __init__(self):
        self=self

        #Split 1 -> biweek = 13 | shift = False
        #Split 2 -> biweek = 13 | shift = True
        #Split 3 -> biweek = 12 | shift = False
        #Split 4 -> biweek = 12 | shift = True
        #Split 5 -> biweek = 11 | shift = False

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

    def train_test_split(self, df_item_per_biweek, train_biweek):
        X_train = df_item_per_biweek[df_item_per_biweek['biweek']<train_biweek].drop(columns=['order'], axis=1)
        X_test = df_item_per_biweek[df_item_per_biweek['biweek']==train_biweek].drop(columns=['order'], axis=1)
        y_train = df_item_per_biweek['order'][df_item_per_biweek['biweek']<train_biweek].values
        y_test = df_item_per_biweek['order'][df_item_per_biweek['biweek']==train_biweek].values
        return X_train, X_test, y_train, y_test
