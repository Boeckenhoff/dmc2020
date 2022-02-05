import pandas as pd
import numpy as np

from preprocessor import *
from featuregenerator import *
from dmcerror import *
from scoregenerator import *
import xgboost as xgb


class XGBModel():

    infos_df = pd.read_csv('D:\\Studium\\aktuelles Semester\\dmc2020\\task\\data\\infos.csv', sep='|')
    items_df = pd.read_csv('D:\\Studium\\aktuelles Semester\\dmc2020\\task\\data\\items.csv', sep='|')
    orders_df = pd.read_csv('D:\\Studium\\aktuelles Semester\\dmc2020\\task\\data\\orders.csv', sep='|')

    promo_in_splits = pd.read_csv('C:\\Users\\janau\\Downloads\\promo_in_splits.csv', index_col=False)

    siip_df = pd.read_csv('D:\\Studium\\aktuelles Semester\\dmc2020\\team_c\\features\\SimilarProductIsInPromotion\\featureSPIIP.csv')


    split = 0
    biweek= 0
    shift= 0
    promo= 0
    log_transform= 0
    pp= 0
    fg= 0
    df_item_per_biweek= 0
    n_rounds= 0
    sg=0


    def __init__(self):
        self=self

    def create_dataframe(self,split,promotion_bin,lags,log_transform=True):

        self.pp = PreProcessor()
        self.fg = FeatureGenerator()

        self.log_transform = log_transform

        if (split==1):
            self.biweek = 13
            self.shift = False
            self.promo = (self.promo_in_splits['promo_01_0.01'].values)*1
        elif (split==2):
            self.biweek = 13
            self.shift = True
            self.promo = (self.promo_in_splits['promo_02_0.01'].values)*1
        elif (split==3):
            self.biweek = 12
            self.shift = False
            self.promo = (self.promo_in_splits['promo_03_0.01'].values)*1
        elif (split==4):
            self.biweek = 12
            self.shift = True
            self.promo = (self.promo_in_splits['promo_04_0.01'].values)*1
        else :
            self.biweek = 11
            self.shift = False
            self.promo = (self.promo_in_splits['promo_05_0.01'].values)*1

        df_item_per_biweek = self.pp.create_item_per_biweek(self.orders_df,self.items_df,self.infos_df, self.promo,self.siip_df,self.shift)

        if (self.log_transform):
            df_item_per_biweek['order'] = np.log1p(df_item_per_biweek['order'].values)

        df_item_per_biweek = self.fg.create_promotion_bin(df_item_per_biweek,promotion_bin['ts1'],promotion_bin['ts2'],promotion_bin['ts3'])

        df_item_per_biweek = self.fg.create_first_time_sold(df_item_per_biweek,self.biweek)

        df_item_per_biweek = self.fg.create_first_time_in_promotion(df_item_per_biweek)

        df_item_per_biweek['priceDiff'] = df_item_per_biweek['recommendedRetailPrice']-df_item_per_biweek['simulationPrice']

        df_item_per_biweek['priceDiffRatio'] = df_item_per_biweek['priceDiff']/df_item_per_biweek['recommendedRetailPrice']

        df_item_per_biweek = self.fg.create_features_to_lag(df_item_per_biweek,['is_promotion','firstTimeInPromotion','firstTimeSold','similarProductsInPromotionCount'],range(1,lags+1))

        self.df_item_per_biweek = df_item_per_biweek

    def start(self, param,n_rounds):
        X_train, X_test, y_train, y_test = self.pp.train_test_split(self.df_item_per_biweek,self.biweek)

        dtrain = xgb.DMatrix(X_train.drop(columns=['itemID','biweek'],axis=1), label=y_train, weight=X_train['simulationPrice'].values)
        dtest = xgb.DMatrix(X_test.drop(columns=['itemID','biweek'],axis=1), label=y_test, weight=X_test['simulationPrice'].values)
        param['nthread'] = -1
        model = xgb.train(          params=param,
                                    num_boost_round=10000,
                                    early_stopping_rounds = 100,
                                    dtrain=dtrain,
                                    obj=dmcerror,
                                    evals=[(dtrain,'train'),(dtest,'test')],
                                    feval=dmcscore_log,
                                    verbose_eval=False)

        print('n_rounds: '+str(model.best_iteration))
        y_pred = model.predict(dtest, ntree_limit=model.best_iteration)
        self.sg = ScoreGenerator(X_test,y_pred,y_test,log_transform=self.log_transform)

        return (self.sg.getMaxScore()-self.sg.getPredictionScore())/self.sg.getMaxScore()*100
