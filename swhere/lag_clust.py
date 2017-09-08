import pandas as pd
from sklearn.cluster import KMeans
from wp_ws import buildLaggedFeatures

power_data = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/all_power.csv", nrows=1000)
wind = pd.read_csv("/Users/xiaofeifei/GitHub/Oxford/clean_wind.csv", nrows=1000)


# add lags of wp for each farm
clust_feat = pd.DataFrame()

for i in xrange(1, 26):
    farm = power_data[power_data["farm"] == i]

    res = buildLaggedFeatures(farm["wp"], lag=6, dropna=False)

    farm = pd.concat([farm, res], axis=1)
    clust_feat = pd.concat([clust_feat, farm], axis=0)

del farm

clust_feat = clust_feat.fillna(0)

# apply kmean cluster of all data
kmean = KMeans(n_clusters=40,random_state=0,max_iter=1,verbose=1).fit_predict(clust_feat[["lag_0", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5"]])

clust_feat["kmean"] = kmean

del kmean

# apply kmean by each individual data
clust = pd.DataFrame()
for i in xrange(1, 26):
    farm = clust_feat[clust_feat["farm"] == i]

    kmean = KMeans(n_clusters=4,random_state=0,max_iter=1).fit_predict(farm[["lag_0","lag_1","lag_2","lag_3","lag_4","lag_5"]])
    kmean = kmean+i*10

    farm["f_kmean"] = kmean
    clust = pd.concat([clust, farm], axis=0)

clust_feat = pd.merge(clust_feat, clust[["id", "f_kmean"]], on='id', how='left')

clust_feat.to_csv("/Users/xiaofeifei/GitHub/Oxford/clust.csv", index =False)









