import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

wp_speed = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/wp_speed.csv")
clust_feat = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/clust.csv")

all_data = pd.merge(wp_speed, clust_feat, on='id', how='left')

del wp_speed, clust_feat

X = all_data.drop("wp", axis=1)
y = all_data[["wp", "fold"]]

# set up 5 fold cv by fold
# predict all data together
n_splits = 5
folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=2016).split(list(set(X["fold"]))))

cv_result = pd.DataFrame()
for j, (train_idx, test_idx) in enumerate(folds):

    fold = list(set(X["fold"]))
    train_idx = [fold[x] for x in list(train_idx)]
    test_idx = [fold[x] for x in list(test_idx)]

    X_train = X.loc[X['fold'].isin(train_idx)]
    y_train = y.loc[y['fold'].isin(train_idx)]
    X_holdout = X.loc[X['fold'].isin(test_idx)]
    y_holdout = y.loc[y['fold'].isin(test_idx)]

    id = X_holdout["id"].values
    y_train = np.array(y_train.drop('fold', axis=1)).ravel()

    X_train=np.array(X_train.drop('fold', axis=1))

    #LightGBM Regressor
    model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=7, learning_rate=0.1, n_estimators=100,
                      max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                      subsample=1, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0,
                      nthread=-1, silent=True)

    model.fit(X_train, y_train)
    X_holdout=np.array(X_holdout.drop('fold', axis=1))
    y_pred = model.predict(X_holdout)[:]

    print ("Model fold %d score %f" % (j, mean_squared_error(y_holdout["wp"], y_pred)))
    y_pred = pd.DataFrame(y_pred)


    id = pd.DataFrame(id)
    ture = y_holdout["wp"].values
    ture = pd.DataFrame(ture)
    y_pred = pd.concat([y_pred, id,ture], axis=1, ignore_index=True)

    cv_result = pd.concat([cv_result, y_pred], axis=0)

cv_result.columns = ['pred', 'id', "ture"]

# print mse
print mean_squared_error(cv_result["ture"], cv_result['pred'])

############################ predict by each individual farms #########################
cv_result_by_farm = pd.DataFrame()
for i in xrange(1, 26):

    farm = all_data[all_data["farm"] == i]

    X = farm.drop("wp", axis=1)
    y = farm[["wp","fold"]]

    n_splits = 5
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=2016).split(list(set(X["fold"]))))

    cv_result = pd.DataFrame()
    for j, (train_idx, test_idx) in enumerate(folds):

        fold = list(set(X["fold"]))
        train_idx = [fold[x] for x in list(train_idx)]
        test_idx = [fold[x] for x in list(test_idx)]

        X_train = X.loc[X['fold'].isin(train_idx)]
        y_train = y.loc[y['fold'].isin(train_idx)]
        X_holdout = X.loc[X['fold'].isin(test_idx)]
        y_holdout = y.loc[y['fold'].isin(test_idx)]

        id = X_holdout["id"].values
        y_train = np.array(y_train.drop('fold', axis=1)).ravel()

        X_train=np.array(X_train.drop('fold', axis=1))

        #LightGBM Regressor
        model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=7, learning_rate=0.1, n_estimators=100,
                          max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                          subsample=1, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0,
                          nthread=-1, silent=True)

        model.fit(X_train, y_train)
        X_holdout=np.array(X_holdout.drop('fold', axis=1))
        y_pred = model.predict(X_holdout)[:]

        print ("Model fold %d score %f" % (j, mean_squared_error(y_holdout["wp"], y_pred)))
        y_pred = pd.DataFrame(y_pred)

        id = pd.DataFrame(id)
        ture = y_holdout["wp"].values
        ture = pd.DataFrame(ture)
        y_pred = pd.concat([y_pred, id,ture], axis=1, ignore_index=True)

        cv_result = pd.concat([cv_result, y_pred], axis=0)

    cv_result.columns = ['pred', 'id', "ture"]
    cv_result_by_farm = pd.concat([cv_result, cv_result_by_farm], axis=0)

# print mse of predition
print mean_squared_error(cv_result_by_farm["ture"], cv_result_by_farm['pred'])

##### ensemble two methods
cv_result_by_farm["pred"] = (cv_result_by_farm['pred'] + cv_result['pred']) / 2

print mean_squared_error(cv_result_by_farm["ture"], cv_result_by_farm['pred'])