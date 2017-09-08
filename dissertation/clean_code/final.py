import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import ElasticNet
from scipy.stats import randint

train = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/GS_data.csv")

train = train.fillna(0)

y = train["Sugars"]

train = train.drop(["Sugars"], axis = 1)

# parameter tuning for lasso
kr = GridSearchCV(Lasso(), cv=5, n_jobs = 6,verbose=1,scoring="neg_mean_squared_error",
                  param_grid={"alpha": [0.001,0.0025,0.005,0.0075,0.01,0115,0.0124,0.0125,0.0126,0.0128,0.0135,
                                        0.015,0.0175,0.02,0.025,0.05,0.075,0.1,1,10]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
result =  pd.DataFrame(kr.cv_results_)
result.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/gs_lasso.csv")

scaler = RobustScaler()
train = scaler.fit_transform(train,y)

# parameter tuning for ridge
kr = GridSearchCV(Ridge(alpha=0.1), cv=5, n_jobs = 6,verbose=1,scoring='neg_mean_squared_error',
                  param_grid={"alpha": [0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.35,
                                        0.5,0.7,1,2,3,5,6,7.5,7,8,9,10,12,15,20,25,30]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_
result = pd.DataFrame(kr.cv_results_)
result.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/gs_ridge.csv")

low= 0.005; high =0.1

# parameter tuning for elastic net
kr = RandomizedSearchCV(ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=5000), n_jobs = 6,cv=5,verbose=1,scoring='neg_mean_squared_error',
                  param_distributions={"alpha": sp_rand(low,high),
                              "l1_ratio": sp_rand(low,high)},n_iter=50)

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_
print pd.DataFrame(kr.cv_results_)
result = pd.DataFrame(kr.cv_results_)
result.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/gs_en.csv")

param_test1 = {
    'max_depth': randint(1,15),
    'min_child_weight': randint(1,15),
}

param_test3 = {
    'reg_alpha':sp_rand(0.005,0.1),
    'gamma':sp_rand(0.01,0.2)
}

param_test4 = {
    'learning_rate':[0.01,0.05,0.07],
    'n_estimators':[100,80,60]
}

dtrain = xgb.DMatrix(data=train, label=y)

params = {
    "max_depth": 7,
    'eta': 0.01,
    # "eval_metric": "rmse",
    "objective": "reg:linear",
    "nthread": 6,
    "base_score": np.mean(y),
    'silent': 1,
    'reg_alpha': 1
}

# parameter tuning for xgboost
kr = GridSearchCV(xgb.XGBRegressor(learning_rate=0.1, n_estimators=32, max_depth=7,
                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                   objective='reg:linear', nthread=4, scale_pos_weight=1, seed=27,reg_alpha=1),
                  param_test4,
                  scoring='neg_mean_squared_error', cv=5, n_jobs=6)

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_
print pd.DataFrame(kr.cv_results_)
result = pd.DataFrame(kr.cv_results_)
result.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/gs_xgb_4.csv")
