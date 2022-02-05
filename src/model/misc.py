import datetime

def dmcscore(y, yhat):
    score = 0
    for i in range(len(y)):
      if y[i]>=yhat[i]:
        score += (y[i]-yhat[i])**2
      else:
        score += ((yhat[i]-y[i])*0.6)**2
    return np.sqrt((score/len(y)))

def dmcscore_xg(yhat, y):
    y = y.get_label()
    yhat = yhat
    return "dmc_score", dmcscore(y, yhat)

def create_features_to_lag(df,features,lags):
  for l in lags:
    df1 = df.copy()
    df1.date = df1.date + datetime.timedelta(days = l)
    df1 = df1[['date','itemID']+features]
    df1.columns = ['date','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
    df = pd.merge(df, df1,on=['date','itemID'],how='left')
  return df


def features_to_lag(df,features,lags):
  for l in range(1,lags,1):
    df1 = df.copy()
    df1.date = df1.date + datetime.timedelta(days = l)
    df1 = df1[['date','itemID']+features]
    df1.columns = ['date','itemID']+ [features_lag+'_lag_'+str(l) for features_lag in features]
    df = pd.merge(df, df1,on=['date','itemID'] ,how='left')
  return df


def feature_agg_order_per_time(df,features,groupby='date'):
  for feature in features:
    for col, agg, aggtype in [('order',np.sum,'sum'),('order',np.mean,'avg')]:
      df2 = df[[feature, groupby, col]].groupby([feature,groupby]).aggregate(agg).reset_index()
      df2.columns = [feature, groupby, feature+'_'+aggtype+'_'+col+'_per_'+groupby]
      df = pd.merge(df, df2, on=[groupby,feature], how='left')
  return df
