import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessor import *
from featuregenerator import *
from dmcerror import *
from scoregenerator import *
import xgboost as xgb


infos_df = pd.read_csv('/content/drive/My Drive/DMC2020/infos.csv', sep='|')
items_df = pd.read_csv('/content/drive/My Drive/DMC2020/items.csv', sep='|')
orders_df = pd.read_csv('/content/drive/My Drive/DMC2020/orders.csv', sep='|')

promo_in_splits = pd.read_csv('/content/drive/My Drive/DMC2020/promo_in_splits.csv', index_col=False)
promo1 = (promo_in_splits['promo_01_0.01'].values)*1
promo2 = (promo_in_splits['promo_02_0.01'].values)*1
promo3 = (promo_in_splits['promo_03_0.01'].values)*1
promo4 = (promo_in_splits['promo_04_0.01'].values)*1
promo5 = (promo_in_splits['promo_05_0.01'].values)*1

siip_df = pd.read_csv('/content/drive/My Drive/DMC2020/featureSPIIP.csv')

pp = PreProcessor()
fg = FeatureGenerator()

#Split 1 -> biweek = 13 | shift = False
#Split 2 -> biweek = 13 | shift = True
#Split 3 -> biweek = 12 | shift = False
#Split 4 -> biweek = 12 | shift = True
#Split 5 -> biweek = 11 | shift = False

biweek=13
shift=False

df_item_per_biweek = pp.create_item_per_biweek(orders_df,items_df,infos_df, promo1,siip_df,shift)

df_item_per_biweek['order'] = np.log1p(df_item_per_biweek['order'].values)

df_item_per_biweek = fg.create_promotion_bin(df_item_per_biweek,95,98,99)

df_item_per_biweek = fg.create_first_time_in_promotion(df_item_per_biweek)

df_item_per_biweek = fg.create_first_time_sold(df_item_per_biweek,biweek)

df_item_per_biweek['priceDiff'] = df_item_per_biweek['recommendedRetailPrice']-df_item_per_biweek['simulationPrice']

df_item_per_biweek['priceDiffRatio'] = df_item_per_biweek['priceDiff']/df_item_per_biweek['recommendedRetailPrice']

df_item_per_biweek = fg.create_features_to_lag(df_item_per_biweek,['is_promotion','firstTimeInPromotion','firstTimeSold','similarProductsInPromotionCount'],[1,2,3])

X_train, X_test, y_train, y_test = pp.train_test_split(df_item_per_biweek,biweek)

dtrain = xgb.DMatrix(X_train.drop(columns=['itemID','biweek','promotion_bin'],axis=1), label=y_train, weight=X_train['simulationPrice'].values)
dtest = xgb.DMatrix(X_test.drop(columns=['itemID','biweek','promotion_bin'],axis=1), label=y_test, weight=X_test['simulationPrice'].values)

bst = xgb.train(params={'disable_default_eval_metric' : 1 ,'colsample_bytree': 0.9, 'eta': 0.02, 'max_depth': 10, 'min_child_weight': 4.0, 'subsample': 0.95},
                num_boost_round=1000,
                dtrain=dtrain,
                #early_stopping_rounds = 100,
                obj=dmcerror,
                evals=[(dtrain,'train'),(dtest,'test')],
                feval=dmcscore_log,
                verbose_eval=True)

y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
sg = ScoreGenerator(X_test,y_pred,y_test,log_transform=True)
sg.getPredictionScore()

sg.getMaxScore()

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(bst, importance_type='gain',  height=0.2, ax=ax, show_values=False)
plt.show
