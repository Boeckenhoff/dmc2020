
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepareData import load_train_test_dates, create_item_perDay, train_test_split
from dmcscore import score
from staticFeatures import one_hot_encoding
from dynamicFeatures import create_mean_order_per_itemID, create_mean_order_per_itemID_per_weekday

path = "D:/Studium/aktuelles Semester/dmc2020/task/data/"

infos_df = pd.read_csv(path+'infos.csv', sep='|')
items_df = pd.read_csv(path+'items.csv', sep='|')
orders_df = pd.read_csv(path+'orders.csv', sep='|')

#train_dates, test_dates = load_train_test_dates(path)


df_item_perDay = create_item_perDay(orders_df,infos_df,items_df)
df_item_perDay[df_item_perDay['itemID']<4].groupby(['week','itemID']).aggregate({'order':np.sum}).unstack().plot(figsize=(20,5), subplots = True)
#df_item_perDay = create_mean_order_per_itemID(df_item_perDay, train_dates[0])
#df_item_perDay = create_mean_order_per_itemID_per_weekday(df_item_perDay, train_dates[0])
#df_item_perDay = one_hot_encoding(df_item_perDay, ['dayofweek'])


#X_train, X_test, y_train, y_test = train_test_split(df_item_perDay, train_dates[0], test_dates[0])


#import xgboost as xgb

#model = xgb.XGBRegressor()
#model.fit(X_train.drop(columns=['date'],axis=1),y_train)
#y_pred = model.predict(X_test.drop(columns=['date'],axis=1))


#score_df = pd.DataFrame({'itemID' : X_test['itemID'].values, 'order' : y_test, 'prediction' : y_pred})