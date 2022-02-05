import pandas as pd
import numpy as np
import datetime

def create_mean_order_per_itemID(df_item_perDay, train_dates):

  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
  df.columns = ['itemID', 'mean_order_per_itemID']
  return pd.merge(df_item_perDay, df, on=['itemID'])

def create_mean_order_per_feature(df_item_perDay, features, train_dates):

  for feat in features:
    df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
    df = df.groupby(feat).aggregate({'order':np.mean}).reset_index()
    df.columns = ['itemID', 'mean_order_per_'+str(feat)]
    df_item_perDay = pd.merge(df_item_perDay, df, on=['itemID'])

  return df_item_perDay

def create_median_order_per_feature(df_item_perDay, features, train_dates):

  for feat in features:
    df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
    df = df.groupby(feat).aggregate({'order':np.median}).reset_index()
    df.columns = ['itemID', 'median_order_per_'+str(feat)]
    df_item_perDay = pd.merge(df_item_perDay, df, on=['itemID'])

  return df_item_perDay

def create_mean_order_per_itemID_per_weekday(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID','dayofweek']).aggregate({'order':np.mean}).unstack()
  df.columns = ['mean_order_per_itemID_per_weekday_'+str(i) for i in range(0,7)]
  df.reset_index()

  return pd.merge(df_item_perDay, df, on='itemID')

def create_median_order_per_itemID_per_weekday(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID','dayofweek']).aggregate({'order':np.median}).unstack()
  df.columns = ['median_order_per_itemID_per_weekday_'+str(i) for i in range(0,7)]
  df.reset_index()

  return pd.merge(df_item_perDay, df, on='itemID')

def create_sold_in_test_period(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
  df['order'] = df['order'].apply(lambda x: 1 if x > 0 else 0)
  df.columns = ['itemID','sold_in_test_period']
  return pd.merge(df_item_perDay, df, on='itemID')

def create_features_to_lag(df,features,lags):
  for l in lags:
    df1 = df.copy()
    df1.date = df1.date + datetime.timedelta(days = l)
    df1 = df1[['date','itemID']+features]
    df1.columns = ['date','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
    df = pd.merge(df, df1,on=['date','itemID'],how='left')
  return df

def create_mean_salesPrice_per_ItemID( df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID']).aggregate({'salesPrice':np.mean}).reset_index()
  df.columns = ['itemID', 'mean_salesPrice_per_itemID']
  return pd.merge(df_item_perDay, df, on=['itemID'])

#Für diese Methode müssen mean_salesPrice_per_ItemID und mean_order_per_itemID im item_perDay df enthalten sein
def create_priceElasticity (df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df['priceElasticity'] = (df['order']/ df['mean_order_per_itemID'] -1 ) / (df['salesPrice']/ df['mean_salesPrice_per_ItemID'] -1 )
  return pd.merge(df_item_perDay, df, on=['itemID'])

#wenn die Order pro ItemID über dem median liegt
def create_high_orderCount(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
  df['order'] = df['order'].apply(lambda x: 1 if x > 135 else 0)
  df.columns = ['itemID','high_orderCount']
  return pd.merge(df_item_perDay, df, on='itemID')

def create_high_simulation_price(df_item_perDay,infos_df,percentile):

  df = infos_df[['itemID','simulationPrice']]
  ts = np.percentile(infos_df['simulationPrice'].values, percentile)
  df['high_simulation_price_'+str(percentile)] = df['simulationPrice'].apply(lambda x: 1 if x > ts  else 0)
  return pd.merge(df_item_perDay, df[['itemID', 'high_simulation_price_'+str(percentile)]], on='itemID')

def create_mean_order_per_feature(df_item_perDay, features, train_dates):
  for feat in features:
    df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
    df = df.groupby(feat).aggregate({'order':np.mean}).reset_index()
    df.columns = [feat, 'mean_order_per_'+str(feat)]
    df_item_perDay = pd.merge(df_item_perDay, df, on=[feat])

  return df_item_perDay

def create_percentile_order_count(df_item_perDay,train_dates):

  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df.groupby(['itemID']).aggregate({'order':np.sum}).reset_index()
  ts25 = np.percentile(df['order'].values, 25)
  ts50 = np.percentile(df['order'].values, 50)
  ts75 = np.percentile(df['order'].values, 75)
  df['percentile_order_count'] = df['order'].apply(lambda x: 1 if (x <= ts25) else 2 if ((x>ts25) & (x<=ts50)) else  3 if ((x>ts50) & (x<=ts75))  else 4)
  return pd.merge(df_item_perDay, df[['itemID', 'percentile_order_count']], on='itemID')

def create_median_orders_base(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df[(df['promotion'] == 0)|(df['promotion'] == False)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
  df.columns = ['itemID','median_order_base']
  return df_item_perDay.merge(df,on = ['itemID'])

def create_mean_orders_base(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df[(df['promotion'] == 0)|(df['promotion'] == False)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
  df.columns = ['itemID','median_order_base']
  return df_item_perDay.merge(df,on = ['itemID'])

def create_median_orders_promotion(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df[(df['promotion'] == 1)|(df['promotion'] == True)].groupby(['itemID']).aggregate({'order':np.median}).reset_index()
  df.columns = ['itemID','median_order_base']
  return df_item_perDay.merge(df,on = ['itemID'])

def create_mean_orders_promotion(df_item_perDay, train_dates):
  df = df_item_perDay[df_item_perDay['date'].isin(train_dates['date'].values)]
  df = df[(df['promotion'] == 1)|(df['promotion'] == True)].groupby(['itemID']).aggregate({'order':np.mean}).reset_index()
  df.columns = ['itemID','median_order_base']
  return df_item_perDay.merge(df,on = ['itemID'])