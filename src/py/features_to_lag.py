import datetime
import pandas as pd

def features_to_lag(df,features,lags):
  for l in range(1,lags,1):
    df1 = df.copy()
    df1.date = df1.date + datetime.timedelta(days = l)
    df1 = df1[['date','itemID']+features]
    df1.columns = ['date','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
    df = pd.merge(df, df1,on=['date','itemID'] ,how='left')
  return df
