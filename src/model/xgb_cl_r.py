import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessor import *
from featuregenerator import *
from dmcscore import score

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = ""

infos_df = pd.read_csv(path+'infos.csv', sep='|')
items_df = pd.read_csv(path+'items.csv', sep='|')
orders_df = pd.read_csv(path+'orders.csv', sep='|')

infos_df = pd.read_csv('/content/drive/My Drive/DMC2020/infos.csv', sep='|')
items_df = pd.read_csv('/content/drive/My Drive/DMC2020/items.csv', sep='|')
orders_df = pd.read_csv('/content/drive/My Drive/DMC2020/orders.csv', sep='|')

pp = PreProcessor()
fg = FeatureGenerator()

evaluation = pd.DataFrame(columns=['split','model','pred-score abs','overfee-score abs', 'dmc-score abs', 'max-score abs','dmc-score %', 'sum order','sum_prediction','runtime','rmse', 'mae', 'hyperparameters', 'features importance'])

df_item_per_biweek = pp.create_item_per_biweek(orders_df,items_df,infos_df)
df_item_per_biweek = pp.create_classification(df_item_per_biweek)
#df_item_per_biweek = pp.create_peak(df_item_per_biweek)
df_item_per_biweek = fg.create_max(df_item_per_biweek,13)
df_item_per_biweek = fg.create_percentile_order_count(df_item_per_biweek,13)
df_item_per_biweek = fg.feature_agg_order_per_time(df_item_per_biweek,['percentile_order_count'],13)
col = list(df_item_per_biweek.columns[13:])
df_item_per_biweek = fg.create_features_to_lag(df_item_per_biweek, col+['order','cl_order'],[1,2])
df_item_per_biweek = df_item_per_biweek.drop(columns=columns, axis=1)
df_item_per_biweek = fg.one_hot_encoding(df_item_per_biweek,['percentile_order_count'])

X_train, X_test, y_train, y_test = pp.train_test_split_cl(df_item_per_biweek, 13)

model = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=99)

model.fit(
    X_train.drop(columns=['biweek','itemID'], axis=1),
    y_train,
    #eval_metric = 'auc',
    eval_set=[(X_train.drop(columns=['biweek','itemID'], axis=1),y_train), (X_test.drop(columns=['biweek','itemID'], axis=1), y_test)],
    verbose=True,
    early_stopping_rounds = 10)

print(classification_report(y_test,model.predict(X_test.drop(columns=['biweek','itemID'], axis=1))))

df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']==13] = model.predict(X_test.drop(columns=['biweek','itemID'], axis=1))

X_train, X_test, y_train, y_test = pp.train_test_split(df_item_per_biweek, 13)
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
    X_train.drop(columns=['biweek','itemID',], axis=1),
    (y_train),
    #eval_metric = 'mae',
    eval_set=[(X_train.drop(columns=['biweek','itemID'], axis=1),(y_train)), (X_test.drop(columns=['biweek','itemID'], axis=1), (y_test))],
    verbose=True,
    early_stopping_rounds = 10)

def score_help (x,y,z):
  if x>y:
    return y*z
  else:
    return x*z-((y-x)*z*0.6)

y_pred = model.predict(X_test.drop(columns=['biweek','itemID'], axis=1))
score_df = X_test.copy()
score_df['order'] = (y_test)
score_df['prediction'] = (y_pred)
score_df['prediction'] = score_df['prediction'].round()
score_df['prediction'] = score_df['prediction'].apply(lambda x : 0 if x<0 else x)
score_df = pd.merge(score_df, infos_df[['itemID','simulationPrice']], on='itemID')
score_df['score'] = score_df.apply(lambda x: score_help(x['order'],x['prediction'],x['simulationPrice']), axis=1)
pred, overfee, maxscore = score( (y_pred), (y_test),infos_df['simulationPrice'].values)

feature_importances = pd.DataFrame({'col': X_test.drop(columns=['biweek','itemID'], axis=1).columns,'imp':model.feature_importances_})
feature_importances = feature_importances.sort_values(by='imp',ascending=False)
feature_importances

evaluation = evaluation.append(pd.Series( ((1),'xgbRegressor',pred,overfee,(pred-overfee),maxscore,((pred-overfee)/maxscore*100),score_df['order'].sum(),score_df['prediction'].sum(),0,np.sqrt(mean_squared_error(y_test,y_pred)), mean_absolute_error(y_test,y_pred),model.get_params,feature_importances), index=evaluation.columns), ignore_index=True)
evaluation

xgb.plot_tree(model, num_trees=2)
fig = plt.gcf()
fig.set_size_inches(150, 100)

score_df.head(20)
