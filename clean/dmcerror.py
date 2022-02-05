import numpy as np

def dmcerror(y_pred,dtrain):
    y_true = dtrain.get_label()
    w = dtrain.get_weight()
    grad = np.where((y_true-y_pred)>0, np.square(w)*(y_pred-y_true), (9*np.square(w)*(y_pred-y_true))/25)
    hess = np.where((y_true-y_pred)>0, np.square(w), (np.square(w)*9)/25)
    return grad, hess

def dmcscore_log(y_pred,dtrain):
  y_true = dtrain.get_label()
  y_pred = np.where((y_pred<0),0,y_pred)
  y_true = np.expm1(y_true)
  y_pred = np.expm1(y_pred)
  w = dtrain.get_weight()
  score = np.where((y_true-y_pred)>0, w*(y_true-y_pred), ((3/5)*w*(y_pred-y_true)))
  return 'dmcscore', float(np.sum(score))

def dmcscore(y_pred,dtrain):
  y_true = dtrain.get_label()
  y_pred = np.where((y_pred<0),0,y_pred)
  w = dtrain.get_weight()
  score = np.where((y_true-y_pred)>0, w*(y_true-y_pred), ((3/5)*w*(y_pred-y_true)))
  return 'dmcscore', float(np.sum(score))