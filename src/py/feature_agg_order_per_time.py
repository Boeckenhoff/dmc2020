import pandas as pd
import numpy as np

def feature_agg_order_per_time(df,features,groupby='date'):
  for feature in features:
    for col, agg, aggtype in [('order',np.sum,'sum'),('order',np.mean,'avg')]:
      df2 = df[[feature, groupby, col]].groupby([feature,groupby]).aggregate(agg).reset_index()
      df2.columns = [feature, groupby, feature+'_'+aggtype+'_'+col+'_per_'+groupby]
      df = pd.merge(df, df2, on=[groupby,feature], how='left')
  return df
