import numpy as np

##as objective=dmcerror
def dmcerror(y_pred,dtrain):
    y_true = dtrain.get_label()
    w = dtrain.get_weight()
    grad = np.where((y_true-y_pred)>0, np.square(w)*(y_pred-y_true), (9*np.square(w)*(y_pred-y_true))/25)
    hess = np.where((y_true-y_pred)>0, np.square(w), (np.square(w)*9)/25)
    return grad, hess

##as eval_metric=dmcscore
def dmcscore(y_pred,dtrain):
  y_true = dtrain.get_label()
  w = dtrain.get_weight()
  score = np.where((y_true-y_pred)>0, w*(y_true-y_pred), ((3/5)*w*(y_pred-y_true)))
  return 'dmcscore', float(np.sum(score))
