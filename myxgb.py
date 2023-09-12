from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import xgboost as xgb
from typing import Tuple
from sklearn.base import BaseEstimator

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label().flatten()
    w = dtrain.get_weight().flatten()
    return w * (predt.flatten() - y)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    w = dtrain.get_weight().flatten()
    return w * np.ones(predt.shape[0])

def weighted_squared(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Weighted Squared Error objective.
    '''
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

def weighted_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    w = dtrain.get_weight()
    elements = w.flatten() * np.power((predt.flatten() - y.flatten()), 2)
    return 'WeightedRMSE', float(np.sqrt(np.mean(elements)))

class MyXGBRegressor(XGBRegressor):
    
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            X, Xval, y, yval = train_test_split(X, y, shuffle=True, test_size=.2)
            super().fit(X, y, eval_set=[(Xval, yval)], verbose=False)
        else:
            X, Xval, y, yval, sample_weight, sample_weight_val = train_test_split(X, y, sample_weight,
                                                                                  shuffle=True, test_size=.2)
            super().fit(X, y, sample_weight=sample_weight,
                        eval_set=[(Xval, yval)], sample_weight_eval_set=[sample_weight_val], verbose=False)
        return self

class MyXGBClassifier(XGBClassifier):
    
    def fit(self, X, y):
        X, Xval, y, yval = train_test_split(X, y, shuffle=True, test_size=.2, stratify=y)
        super().fit(X, y, eval_set=[(Xval, yval)], verbose=False)
        return self

class MyWeightedΧGBRegressor(BaseEstimator):
    
    def __init__(self):
        return
    def fit(self, X, y, *, sample_weight):
        X, Xval, y, yval, sample_weight, sample_weight_val = train_test_split(X, y, sample_weight,
                                                                              shuffle=True, test_size=.2)
        dtrain = xgb.DMatrix(X, y, weight=sample_weight)
        dval = xgb.DMatrix(Xval, yval, weight=sample_weight_val)
        self.model = xgb.train({'max_depth': 2, 'learning_rate': .05, 'min_child_weight': 20,
                                'disable_default_eval_metric': 1, 'seed': 123},
                              dtrain, num_boost_round=500, evals=[(dval, 'val')], obj=weighted_squared,
                              custom_metric=weighted_metric, early_stopping_rounds=5, verbose_eval=False)
        return self

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))

xgb_reg = lambda: MyXGBRegressor(max_depth=2, learning_rate=.05, n_estimators=500,
                             early_stopping_rounds=5, min_child_weight=20,
                             verbosity=0, random_state=123)
xgb_clf = lambda: MyXGBClassifier(max_depth=2, learning_rate=.05, n_estimators=500,
                              early_stopping_rounds=5, min_child_weight=20, verbosity=0,
                              random_state=123)
xgb_wreg = lambda: MyWeightedΧGBRegressor()