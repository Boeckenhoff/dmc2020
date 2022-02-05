import numpy as np
import pandas as pd

class ScoreGenerator():

    score_df=pd.DataFrame()

    def __init__(self,X_test,y_pred,y_test,log_transform):
        self=self
        self.X_test = X_test
        if (log_transform):
            self.y_pred = np.expm1(y_pred)
            self.y_test = np.expm1(y_test)
        else:
            self.y_pred = y_pred
            self.y_test = y_test

        def score_help (x,y,z):
          if x>=y:
            return y*z
          else:
            return x*z-((y-x)*z*0.6)

        self.score_df = self.X_test.copy()
        self.score_df['order'] = self.y_test
        self.score_df['prediction'] = self.y_pred
        self.score_df['prediction'] = self.score_df['prediction'].round()
        self.score_df['prediction'] = self.score_df['prediction'].apply(lambda x : 0 if x<0 else x)
        self.score_df['score'] = self.score_df.apply(lambda x: score_help(x['order'],x['prediction'],x['simulationPrice']), axis=1)
        self.score_df['max_score'] = self.score_df['order'] * self.score_df['simulationPrice']
        self.score_df['diff'] = self.score_df['max_score'] - self.score_df['score']

    def getPredictionScore(self):
        return self.score_df['score'].sum()

    def getMaxScore(self):
        return self.score_df['max_score'].sum()

    def getPredictionDf(self):
        pred_df = self.score_df[['itemID','prediction']]
        pred_df.rename(columns={'prediction' : 'demandPrediction'}, inplace=True)
        pred_df = pred_df.astype('int')
        return pred_df
