import re
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLarsCV, Lasso
from sklearn.model_selection import KFold
import csv

class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, y)

    def predict(self, X):
        return super(LogExpPipeline, self).predict(X)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


train = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_feature.csv")
train = train.fillna(0)

y = train[['Sugars']]

train_data_features = train.drop(['Sugars',"micro_num"], axis=1)

S_train = np.zeros((20, 2))

# 20 runs of 5-fold cross-validation
for i in xrange(20):
    folds = list(KFold(n_splits=2, shuffle=True, random_state=i).split(train_data_features, y))

    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = train_data_features.loc[train_idx]
        y_train = y.loc[train_idx]
        X_holdout = train_data_features.loc[test_idx]
        y_holdout = y.loc[test_idx]

        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtest = xgb.DMatrix(data=X_holdout)

        params = {
            "max_depth": 7,
            'eta': 0.1,
            'reg_alpha': 1,
            "eval_metric": "rmse",
            "objective": "reg:linear",
            "nthread": 6,
            "base_score": np.mean(y_train["Sugars"]),
            'silent': 1
        }

        model = xgb.train(params, dtrain=dtrain, num_boost_round=40)
        pred_xgb = model.predict(dtest)

        ######## svm ###########
        svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SVR(kernel='rbf', C=30, epsilon=0.05)]))
        svm_pipe.fit(X_train, y_train)
        pred_svm = svm_pipe.predict(X_holdout)[:]


        ############### en #############

        en = ElasticNet(alpha=0.005, l1_ratio=0.5)
        en.fit(X_train, y_train)
        pred_en = en.predict(X_holdout)[:]


        ############ lasso ############
        lasso = Lasso(normalize=True,alpha=0.01)
        lasso.fit(X_train, y_train)
        pred_la = lasso.predict(X_holdout)[:]

        ############ ensemble ###########
        pred = (pred_xgb+pred_svm+pred_en+pred_la)/4

        mse = mean_squared_error(y_holdout, pred)


        S_train[i, j] = mse


S_train = pd.DataFrame(S_train)
S_train.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/result_model7.csv")

