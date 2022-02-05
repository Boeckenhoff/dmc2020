from xgb_model import *
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials, Trials


def score(params):


    n_rounds = int(params['n_rounds'])
    del params['n_rounds']
    
    print("Training with params : ")
    print(params)
    
    score = 0

    for i in range(1,6):

      promotion_bin = { 'ts1':95,'ts2':98,'ts3':99 }
      lags = 3
      xgb = XGBModel()
      xgb.create_dataframe(i,promotion_bin,lags)
      score=score+xgb.start(params,n_rounds)

   
    print("\tScore {0}\n\n".format(score/5))

    return {'loss': score/5, 'status': STATUS_OK}


def optimize(trials):
  space = {
              'n_rounds' : hp.quniform('n_rounds', 100, 2500, 1),
              'eta': hp.quniform('eta', 0.01, 0.1, 0.01),
              'max_depth':  hp.choice('max_depth', np.arange(3, 25, dtype=int)),
              'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1),
              'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
              'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
              'reg_alpha' : hp.quniform('reg_alpha', 0, 4, 0.05),
              'reg_lambda' : hp.quniform('reg_lambda', 0, 4, 0.05)
              }

  best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=500)

  print(best)



#trials = SparkTrials(parallelism=4)
trials = Trials()

optimize(trials)
