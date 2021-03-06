import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# load in data
power_data = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/all_power.csv", nrows=1000000)
wind = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/clean_wind.csv", nrows=1000)

# add lags function
def buildLaggedFeatures(s, lag=2, dropna=True):
    if type(s) is pd.DataFrame:
        new_dict = {}
        for col_name in s:
            new_dict[col_name] = s[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=s.index)

    elif type(s) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([s.shift(i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print 'Only works for DataFrame or Series'
        return None
    if dropna:
        return res.dropna()
    else:
        return res

# add fold
time = int(np.round(power_data.shape[0] / float(600)))
fold = np.repeat(np.arange(1, time + 1, 1), 600)
power_data["fold"] = fold[1:power_data.shape[0] + 1]

del time, fold

# predict wp by ws
wind = pd.merge(wind, power_data, on='date', how='left')

del power_data

# create new variables w2 = ws^2, w3 = ws^3
wind["ws1"] = wind["windspeed_forecast"]
wind["ws2"] = np.square(wind["ws1"])
wind["ws3"] = np.power(wind["ws1"], 3)
wind.drop(["windspeed_forecast"], axis = 1)

X = wind[["ws1", "ws2", "ws3","fold","id"]]
y = wind[["wp","fold"]]

# 5 fold cv, split by fold
n_splits = 5
folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=2016).split(list(set(X["fold"]))))

# save the pred result
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
    farm = X_holdout['farm'].values

    y_train = np.array(y_train.drop('fold', axis=1)).ravel()

    X_train=np.array(X_train.drop('fold', axis=1))

    #LightGBM Regressor
    model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=8, learning_rate=0.1, n_estimators=675,
                      max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10,
                      subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0,
                      nthread=-1, silent=True)

    model.fit(X_train, y_train)
    X_holdout=np.array(X_holdout.drop('fold', axis=1))
    y_pred = model.predict(X_holdout)[:]

    print ("Model fold %d score %f" % (j, mean_squared_error(y_holdout["wp"], y_pred)))
    y_pred = pd.DataFrame(y_pred)

    id = pd.DataFrame(id)
    farm = pd.DataFrame(farm)
    y_pred = pd.concat([y_pred, id, farm], axis=1, ignore_index=True)

    cv_result = pd.concat([cv_result, y_pred], axis=0)

    del id, farm, y_pred


# rename the columns
cv_result.columns = ['wp_pred', 'id', 'farm']
cv_result = cv_result.sort(['id'], ascending=[1])

# add 3 lags of predict wp by farm
wp_ws = pd.DataFrame()

for i in xrange(1, 26):
    farm = cv_result[cv_result["farm"] == i]
    res = buildLaggedFeatures(farm["wp_pred"], lag=6, dropna=False)
    farm = pd.concat([farm, res], axis=1)
    wp_ws = pd.concat([wp_ws, farm], axis=0)



wp_ws.to_csv("/Users/xiaofeifei/GitHub/Oxford/wp_speed.csv", index =False)