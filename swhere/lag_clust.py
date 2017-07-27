
import pandas as pd
from sklearn.cluster import KMeans

power_data = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/all_power.csv", nrows=1000)
wind = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/clean_wind.csv", nrows=1000)


# add lags
def buildLaggedFeatures(s, lag=2, dropna=True):
    if type(s) is pd.DataFrame:
        new_dict = {}
        for col_name in s:
            new_dict[col_name] = s[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(-l)
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


lag_feat = pd.DataFrame()

for i in xrange(1, 26):
    farm = power_data[power_data["farm"] == i]

    res = buildLaggedFeatures(farm["wp"], lag=6, dropna=False)

    farm = pd.concat([farm, res], axis=1)
    lag_feat = pd.concat([lag_feat, farm], axis=0)

del farm

lag_feat = lag_feat.fillna(0)

kmean = KMeans(n_clusters=40,random_state=0,max_iter=1,verbose=1).fit_predict(lag_feat[["lag_0","lag_1","lag_2","lag_3","lag_4","lag_5"]])

lag_feat["kmean"] = kmean

clust = pd.DataFrame()
for i in xrange(1, 26):
    farm = lag_feat[lag_feat["farm"] == i]

    kmean = KMeans(n_clusters=4,random_state=0,max_iter=1).fit_predict(farm[["lag_0","lag_1","lag_2","lag_3","lag_4","lag_5"]])
    kmean = kmean+i*10

    farm["f_kmean"] = kmean
    clust = pd.concat([clust, farm], axis=0)

lag_feat = pd.merge(lag_feat, clust[["id","f_kmean"]], on='id', how='left')

lag_feat.to_csv("/Users/xiaofeifei/GitHub/Oxford/clust.csv", index =False)









