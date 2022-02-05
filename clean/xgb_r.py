import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessor import *
from featuregenerator import *
from dmcerror import *
from scoregenerator import *
import xgboost as xgb


from google.colab import drive
drive.mount('/content/drive')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = ""

infos_df = pd.read_csv(path+'infos.csv', sep='|')
items_df = pd.read_csv(path+'items.csv', sep='|')
orders_df = pd.read_csv(path+'orders.csv', sep='|')
promo_df = pd.read_csv(path+'promotiondates.csv')
siip_df = pd.read_csv(path+'featureSPIIP.csv')

pp = PreProcessor()
fg = FeatureGenerator()

biweek=13

df_item_per_biweek = pp.create_item_per_biweek(orders_df,items_df,infos_df, promo_df,siip_df)
df_item_per_biweek['order'] = np.log1p(df_item_per_biweek['order'].values)
df_item_per_biweek = pp.create_promotion_bin(df_item_per_biweek,75,95,99)
df_item_per_biweek = fg.create_features_to_lag(df_item_per_biweek, ['order','is_promotion'],[1,2,3])
df_item_per_biweek = fg.create_first_time_in_promotion(df_item_per_biweek)
df_item_per_biweek = fg.create_first_time_sold(df_item_per_biweek,biweek)
df_item_per_biweek['priceDiff'] = df_item_per_biweek['recommendedRetailPrice']-df_item_per_biweek['simulationPrice']
df_item_per_biweek['priceDiffRatio'] = df_item_per_biweek['priceDiff']/df_item_per_biweek['recommendedRetailPrice']

X_train, X_test, y_train, y_test = pp.train_test_split(df_item_per_biweek,biweek)

dtrain = xgb.DMatrix(X_train.drop(columns=['itemID','biweek'],axis=1), label=y_train, weight=X_train['simulationPrice'].values)
dtest = xgb.DMatrix(X_test.drop(columns=['itemID','biweek'],axis=1), label=y_test, weight=X_test['simulationPrice'].values)

bst = xgb.train(params={'disable_default_eval_metric' : 1,'colsample_bytree': 0.85, 'eta': 0.05, 'max_depth': 12, 'min_child_weight': 2.0, 'subsample': 0.8},
                num_boost_round=300,
                dtrain=dtrain,
                #early_stopping_rounds = 20,
                obj=dmcerror,
                evals=[(dtrain,'train'),(dtest,'test')],
                feval=dmcscore,
                verbose_eval=True)

y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
sg = ScoreGenerator(X_test,y_pred,y_test,log_transform=True)
sg.getPredictionScore()
