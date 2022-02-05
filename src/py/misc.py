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
