import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessor import *
from featuregenerator import *
from dmcerror import *
from scoregenerator import *
import xgboost as xgb

import xgboost as xgb


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = ""

infos_df = pd.read_csv(path+'infos.csv', sep='|')
items_df = pd.read_csv(path+'items.csv', sep='|')
orders_df = pd.read_csv(path+'orders.csv', sep='|')
promo_df = pd.read_csv(path+'promotiondates.csv')

pp = PreProcessor()
fg = FeatureGenerator()

biweek=13

df_item_per_biweek = pp.create_item_per_biweek(orders_df,items_df,infos_df, promo_df)
df_item_per_biweek = pp.create_classification_bins(df_item_per_biweek,biweek)

df_item_per_biweek = fg.create_max(df_item_per_biweek,biweek)
df_item_per_biweek = fg.create_percentile_order_count(df_item_per_biweek,biweek)
df_item_per_biweek = fg.create_features_to_lag(df_item_per_biweek, ['order','cl_order','is_promotion'],[1,2,3])
df_item_per_biweek = fg.one_hot_encoding(df_item_per_biweek,['percentile_order_count'])
df_item_per_biweek = fg.create_mean_orders_base(df_item_per_biweek, biweek)
df_item_per_biweek = fg.create_median_orders_base(df_item_per_biweek, biweek)
df_item_per_biweek = fg.create_mean_orders_promotion(df_item_per_biweek, biweek)
df_item_per_biweek = fg.create_median_orders_promotion(df_item_per_biweek, biweek)

df_item_per_biweek = pd.merge(df_item_per_biweek, infos_df[['itemID','simulationPrice']], on='itemID')

X_train, X_test, y_train, y_test = pp.train_test_split_cl(df_item_per_biweek, biweek)

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
    eval_set=[(X_train.drop(columns=['biweek','itemID'], axis=1),y_train), (X_test.drop(columns=['biweek','itemID'], axis=1), y_test)],
    verbose=True,
    early_stopping_rounds = 20)

df_item_per_biweek['cl_order'][df_item_per_biweek['biweek']==biweek] = model.predict(X_test.drop(columns=['biweek','itemID'], axis=1))

X_train, X_test, y_train, y_test = pp.train_test_split(df_item_per_biweek, biweek)

dtrain = xgb.DMatrix(X_train.drop(columns=['biweek','itemID'], axis=1), label=y_train, weight=X_train['simulationPrice'].values)
dtest = xgb.DMatrix(X_test.drop(columns=['biweek','itemID'], axis=1), label=y_test, weight=X_test['simulationPrice'].values)

bst = xgb.train(params={'eta' : 0.3, 'reg_lambda' : 0, 'reg_alpha' : 0, 'disable_default_eval_metric' : 1},
                num_boost_round=1000,
                dtrain=dtrain,
                early_stopping_rounds = 20,
                obj=dmcerror,
                evals=[(dtrain,'train'),(dtest,'test')],
                feval=dmcscore,
                verbose_eval=True)

y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)

sg = ScoreGenerator(X_test,y_pred,y_test,log_transform=False)

sg.getPredictionDf().to_csv('team_d.csv', sep='|', index=False, )
