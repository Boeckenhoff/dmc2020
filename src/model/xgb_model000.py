import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from google.colab import drive
drive.mount('/content/drive')

from prepareData import load_train_test_dates, create_item_perDay, train_test_split
from dynamicFeatures import create_high_simulation_price, create_high_orderCount, create_percentile_order_count
from dmcscore import score
from staticFeatures import one_hot_encoding

import xgboost as xgb
import time as t
from sklearn.metrics import mean_squared_error, mean_absolute_error

%matplotlib inline

path = ""

infos_df = pd.read_csv(path+'infos.csv', sep='|')
items_df = pd.read_csv(path+'items.csv', sep='|')
orders_df = pd.read_csv(path+'orders.csv', sep='|')
train_dates, test_dates = load_train_test_dates(path)

evaluation = pd.DataFrame(columns=['split','model','pred-score abs','overfee-score abs', 'dmc-score abs', 'max-score abs','dmc-score %', 'sum order','sum_prediction','runtime','rmse', 'mae', 'hyperparameters', 'features importance'])

for i in range(0,11):


  df_item_perDay = create_item_perDay(orders_df,infos_df,items_df)
  df_item_perDay = create_high_simulation_price(df_item_perDay, infos_df, 75)
  df_item_perDay = create_percentile_order_count(df_item_perDay, train_dates[i])
  df_item_perDay = one_hot_encoding(df_item_perDay, ['dayofweek','percentile_order_count'])
  df_item_perDay = pd.merge(df_item_perDay, infos_df[['itemID','simulationPrice']], on='itemID', how='left')


  start = t.time()
  X_train, X_test, y_train, y_test = train_test_split(df_item_perDay[[
                                                                      'itemID',
                                                                      'date',
                                                                      #'week',
                                                                      'order',
                                                                      #'day',
                                                                      #'month',
                                                                      #'dateInt',
                                                                      'brand',
                                                                      'manufacturer',
                                                                      'customerRating',
                                                                      'category1',
                                                                      'category2',
                                                                      'category3',
                                                                      #'recommendedRetailPrice',
                                                                      #'high_simulation_price_75',
                                                                      'simulationPrice',
                                                                      #'dayofweek_0',
                                                                      #'dayofweek_1',
                                                                      #'dayofweek_2',
                                                                      #'dayofweek_3',
                                                                      #'dayofweek_4',
                                                                      #'dayofweek_5',
                                                                      #'dayofweek_6',
                                                                      #'percentile_order_count_1',
                                                                      #'percentile_order_count_2',
                                                                      #'percentile_order_count_3',
                                                                      #'percentile_order_count_4',
                                                                      ]], train_dates[i], test_dates[i])

  model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=99)

  model.fit(
    X_train.drop(columns=['date'],axis=1),
    y_train,
    #eval_metric = 'mae',
    eval_set=[(X_train.drop(columns=['date'],axis=1), y_train), (X_test.drop(columns=['date'],axis=1), y_test)],
    verbose=True,
    early_stopping_rounds = 10)

  y_pred = model.predict(X_test.drop(columns=['date'],axis=1))
  df = pd.DataFrame({'itemID' : X_test['itemID'].values, 'date' : X_test['date'].values, 'order' : (y_test), 'prediction' : (y_pred)})
  score_df = df.groupby(['itemID']).aggregate({'order':np.sum,'prediction':np.sum})
  score_df = score_df.round()
  score_df[score_df['order']<0] = 0
  score_df = pd.merge(score_df, infos_df[['itemID','simulationPrice']], on='itemID')
  date_df = df.groupby(['date']).aggregate({'order':np.sum,'prediction':np.sum})
  pred,overfee,maxscore = score(score_df['prediction'].values, score_df['order'].values, infos_df['simulationPrice'].values)

  feature_importances = pd.DataFrame({'col': X_test.drop(columns=['date'],axis=1).columns,'imp':model.feature_importances_})
  feature_importances = feature_importances.sort_values(by='imp',ascending=False)

  time = range(1,15)
  order = date_df['order'].values
  preds = date_df['prediction'].values
  plt.plot(time, order, label='order')
  plt.plot(time, preds, label='predictions')
  plt.legend()
  plt.show()

  end = t.time()
  dur = (end-start)/60

  evaluation = evaluation.append(pd.Series( ((i+1),'xgbRegressor',pred,overfee,(pred-overfee),maxscore,((pred-overfee)/maxscore*100),score_df['order'].sum(),score_df['prediction'].sum(),dur,np.sqrt(mean_squared_error(y_test,y_pred)), mean_absolute_error(y_test,y_pred),model.get_params,feature_importances), index=evaluation.columns), ignore_index=True)

evaluation.to_csv('eval.csv')
