import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import RobustScaler

from xgb import XG
import numpy as np

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')

y = train["y"]

train = train.drop(["y"], axis = 1)
params = {
    "max_depth": 8,
    'eta': 0.05,
    'n_trees': 1000,
    "eval_metric": "mse",
    "objective": "reg:linear",
    "nthread": 6,
    "base_score": np.mean(y["Sugars"]),
    "early_stopping_rounds": 10,
    'silent': 1
}

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.(**ind_params),
                            cv_params,
                             scoring = 'accuracy', cv = 5, n_jobs = -1)



kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_
print kr.cv_results_
# {'alpha': 0.01, 'l1_ratio': 0.9}
# 0.543871287359