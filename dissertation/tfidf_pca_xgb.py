import re
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLarsCV,Lasso


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

train1 = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train1 = train1.fillna(0)
train1 = train1.drop(["class"], axis=1)

main_sub = ["CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45']
#
# main_sub = ["CEREAL"]

result1 = {}
result2 = {}
result3 = {}
for sub in main_sub:
    print sub

    train = train1.loc[train1['SubDeptCode'] == sub]

    if train.shape[0]<=200:
        continue

    train = train.reset_index(drop=True)
    def review_to_words( raw_review ):
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review).get_text()
        #
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return( " ".join( meaningful_words ))

    # Get the number of reviews based on the dataframe column size
    num_reviews = train["SkuName"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    for i in xrange( 0, num_reviews ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append( review_to_words(train["SkuName"][i] ) )

    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange( 0, num_reviews ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % ( i+1, num_reviews )
        clean_train_reviews.append( review_to_words( train["SkuName"][i] ))

    print "Creating the bag of words...\n"

    model1 = []
    model2 = []
    model3 = []

    for i in xrange(20):

        X_train, X_valid, y_train, y_valid = train_test_split(clean_train_reviews, train["Sugars"], test_size=0.2, random_state=i)
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.

        train2 = train.ix[list(y_train.index)]
        test2 = train.ix[list(y_valid.index)]

        train2 = train2.reset_index(drop=True)
        test2 = test2.reset_index(drop=True)

        word_len = []
        for item in X_train:
            word_len.append(len(item.split()))

        train2["word_count"] = word_len

        word_len = []
        for item in X_valid:
            word_len.append(len(item.split()))

        test2["word_count"] = word_len

        train2 = train2.join(train2.groupby('MicroDeptCode')['Sugars'].mean(), on='MicroDeptCode', rsuffix='_mean')
        test2 = test2.join(train2.groupby('MicroDeptCode')['Sugars'].mean(), on='MicroDeptCode', rsuffix='_mean')

        train2 = train2.join(train2.groupby('MicroDeptCode')['Sugars'].var(), on='MicroDeptCode', rsuffix='_var')
        test2 = test2.join(train2.groupby('MicroDeptCode')['Sugars'].var(), on='MicroDeptCode', rsuffix='_var')

        train2 = train2.join(train2.groupby('MicroDeptCode')['Sugars'].min(), on='MicroDeptCode', rsuffix='_min')
        test2 = test2.join(train2.groupby('MicroDeptCode')['Sugars'].min(), on='MicroDeptCode', rsuffix='_min')

        train2 = train2.join(train2.groupby('MicroDeptCode')['Sugars'].max(), on='MicroDeptCode', rsuffix='_max')
        test2 = test2.join(train2.groupby('MicroDeptCode')['Sugars'].max(), on='MicroDeptCode', rsuffix='_max')
        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,
                                     ngram_range=(1,2), max_features=100)
        # one way
        train2 = train2.join(train2.groupby(['code1'])['Sugars'].mean(), on=['code1'], rsuffix='_mean1')
        test2 = test2.join(train2.groupby(['code1'])['Sugars'].mean(), on=['code1'], rsuffix='_mean1')

        train2 = train2.join(train2.groupby('code1')['Sugars'].var(), on='code1', rsuffix='_var1')
        test2 = test2.join(train2.groupby('code1')['Sugars'].var(), on='code1', rsuffix='_var1')

        train2 = train2.join(train2.groupby('code1')['Sugars'].min(), on='code1', rsuffix='_min1')
        test2 = test2.join(train2.groupby('code1')['Sugars'].min(), on='code1', rsuffix='_min1')

        train2 = train2.join(train2.groupby('code1')['Sugars'].max(), on='code1', rsuffix='_max1')
        test2 = test2.join(train2.groupby('code1')['Sugars'].max(), on='code1', rsuffix='_max1')

        # # two way inter
        # train2 = train2.join(train2.groupby(['code1',"code2"])['Sugars'].mean(), on=['code1',"code2"], rsuffix='_mean12')
        # train2 = train2.join(train2.groupby(['code1',"code3"])['Sugars'].mean(), on=['code1',"code3"], rsuffix='_mean13')
        # train2 = train2.join(train2.groupby(['code2',"code3"])['Sugars'].mean(), on=['code2',"code3"], rsuffix='_mean23')
        #
        # test2 = test2.join(train2.groupby(['code1',"code2"])['Sugars'].mean(), on=['code1',"code2"], rsuffix='_mean12')
        # test2 = test2.join(train2.groupby(['code1',"code3"])['Sugars'].mean(), on=['code1',"code3"], rsuffix='_mean13')
        # test2 = test2.join(train2.groupby(['code2',"code3"])['Sugars'].mean(), on=['code2',"code3"], rsuffix='_mean23')
        #
        # train2 = train2.join(train2.groupby(['code1',"code2"])['Sugars'].var(), on=['code1',"code2"], rsuffix='_var12')
        # train2 = train2.join(train2.groupby(['code1',"code3"])['Sugars'].var(), on=['code1',"code3"], rsuffix='_var13')
        # train2 = train2.join(train2.groupby(['code2',"code3"])['Sugars'].var(), on=['code2',"code3"], rsuffix='_var23')
        #
        # test2 = test2.join(train2.groupby(['code1',"code2"])['Sugars'].var(), on=['code1',"code2"], rsuffix='_var12')
        # test2 = test2.join(train2.groupby(['code1',"code3"])['Sugars'].var(), on=['code1',"code3"], rsuffix='_var13')
        # test2 = test2.join(train2.groupby(['code2',"code3"])['Sugars'].var(), on=['code2',"code3"], rsuffix='_var23')
        #
        # train2 = train2.join(train2.groupby(['code1',"code2"])['Sugars'].min(), on=['code1',"code2"], rsuffix='_min12')
        # train2 = train2.join(train2.groupby(['code1',"code3"])['Sugars'].min(), on=['code1',"code3"], rsuffix='_min13')
        # train2 = train2.join(train2.groupby(['code2',"code3"])['Sugars'].min(), on=['code2',"code3"], rsuffix='_min23')
        #
        # test2 = test2.join(train2.groupby(['code1',"code2"])['Sugars'].min(), on=['code1',"code2"], rsuffix='_min12')
        # test2 = test2.join(train2.groupby(['code1',"code3"])['Sugars'].min(), on=['code1',"code3"], rsuffix='_min13')
        # test2 = test2.join(train2.groupby(['code2',"code3"])['Sugars'].min(), on=['code2',"code3"], rsuffix='_min23')
        #
        # train2 = train2.join(train2.groupby(['code1',"code2"])['Sugars'].max(), on=['code1',"code2"], rsuffix='_max12')
        # train2 = train2.join(train2.groupby(['code1',"code3"])['Sugars'].max(), on=['code1',"code3"], rsuffix='_max13')
        # train2 = train2.join(train2.groupby(['code2',"code3"])['Sugars'].max(), on=['code2',"code3"], rsuffix='_max23')
        #
        # test2 = test2.join(train2.groupby(['code1',"code2"])['Sugars'].max(), on=['code1',"code2"], rsuffix='_max12')
        # test2 = test2.join(train2.groupby(['code1',"code3"])['Sugars'].max(), on=['code1',"code3"], rsuffix='_max13')
        # test2 = test2.join(train2.groupby(['code2',"code3"])['Sugars'].max(), on=['code2',"code3"], rsuffix='_max23')

        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,
                                     ngram_range=(1,1), max_features=100)
        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data_features = vectorizer.fit_transform(X_train)


        # Numpy arrays are easy to work with, so convert the result to an
        # array

        test_data_features = vectorizer.transform(X_valid)
        test_data_features = test_data_features.toarray()
        test_data_features = pd.DataFrame(test_data_features)

        # test_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/test_1.csv", index=False)
        train_data_features = train_data_features.toarray()
        train_data_features = pd.DataFrame(train_data_features)
        # train_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_1.csv",index=False)

        y_train = pd.DataFrame(y_train)

        # y_train.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_y.csv",index=False)

        y_valid = pd.DataFrame(y_valid)
        # y_valid.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/test_y.csv",index=False)

        n_comp = 15

        # PCA
        pca = PCA(n_components=n_comp, random_state=420)
        pca2_results_train = pca.fit_transform(train_data_features)
        pca2_results_test = pca.transform(test_data_features)

        # # tSVD
        # tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
        # tsvd_results_train = tsvd.fit_transform(train_data_features.drop(["y"], axis=1))
        # tsvd_results_test = tsvd.transform(test_data_features)
        #
        # # ICA
        # ica = FastICA(n_components=n_comp, random_state=420)
        # ica2_results_train = ica.fit_transform(train_data_features.drop(["y"], axis=1))
        # ica2_results_test = ica.transform(test_data_features)
        #
        # # GRP
        # grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
        # grp_results_train = grp.fit_transform(train_data_features.drop(["y"], axis=1))
        # grp_results_test = grp.transform(test_data_features)
        #
        # # SRP
        # srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
        # srp_results_train = srp.fit_transform(train_data_features.drop(["y"], axis=1))
        # srp_results_test = srp.transform(test_data_features)

        for i in range(1, n_comp+1):
            train_data_features['pca_' + str(i)] = pca2_results_train[:,i-1]
            test_data_features['pca_' + str(i)] = pca2_results_test[:, i-1]



        train_data_features = pd.concat([train_data_features, train2[[
        "Sugars_mean", "Sugars_min","Sugars_max", "Sugars_var",
            "code1", "code2", "code3",
        # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
        # "Sugars_var12","Sugars_var13","Sugars_var23",
        # "Sugars_max12","Sugars_max13","Sugars_max23",
        # "Sugars_min12","Sugars_min13","Sugars_min23",
        "Sugars_mean1", "Sugars_min1","Sugars_max1", "Sugars_var1",
                                                "SkuCode"]]], axis=1)

        test_data_features = pd.concat([test_data_features, test2[[
        "Sugars_mean", "Sugars_min","Sugars_max", "Sugars_var",
            "code1", "code2", "code3",
        # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
        # "Sugars_var12","Sugars_var13","Sugars_var23",
        # "Sugars_max12","Sugars_max13","Sugars_max23",
        # "Sugars_min12","Sugars_min13","Sugars_min23",
        "Sugars_mean1", "Sugars_min1","Sugars_max1", "Sugars_var1",
                                        "SkuCode"]]], axis=1)

        feature_names = list(train_data_features.columns.values)

        create_feature_map(feature_names)


        dtrain = xgb.DMatrix(data =train_data_features, label = y_train)
        dtest = xgb.DMatrix(data =test_data_features)

        params = {
                "max_depth": 7,
                'eta': 0.1,
                'reg_alpha':1,
                "eval_metric": "rmse",
                "objective": "reg:linear",
                "nthread" : 6,
                "base_score" : np.mean(y_train["Sugars"]),
                'silent': 1
        }

        model = xgb.train(params, dtrain=dtrain,num_boost_round=40)
        pred = model.predict(dtest)

        S_train = np.zeros((test_data_features.shape[0], 4))

        S_train[np.arange(test_data_features.shape[0]), 0] = pred.ravel()

        ######## svm ###########
        svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SVR(kernel='rbf', C=30, epsilon=0.05)]))
        train_data_features = train_data_features.fillna(0)
        test_data_features = test_data_features.fillna(0)
        svm_pipe.fit(train_data_features, y_train)
        pred = svm_pipe.predict(test_data_features)[:]

        S_train[np.arange(test_data_features.shape[0]), 1] = pred.ravel()

        ############### en #############57.9964231533

        en = ElasticNet(alpha=0.005, l1_ratio=0.5)
        en.fit(train_data_features, y_train)
        pred = en.predict(test_data_features)[:]

        S_train[np.arange(test_data_features.shape[0]), 2] = pred.ravel()

        ############ lasso ############ 56.8123535273
        lasso = Lasso(normalize=True,alpha=0.01)
        lasso.fit(train_data_features, y_train)
        pred = lasso.predict(test_data_features)[:]

        S_train[np.arange(test_data_features.shape[0]), 3] = pred.ravel()

        mse = mean_squared_error(y_valid,S_train.mean(axis=1))
        model2.append(mse)

        mse = mean_squared_error(y_valid, np.repeat(np.mean(y_train["Sugars"]),y_valid.shape[0]))
        model1.append(mse)

        micro_train = train.ix[list(y_train.index),["MicroDeptCode", "Sugars"]]
        micro_test = train.ix[list(y_valid.index),["MicroDeptCode", "Sugars"]]

        micro_mean =  micro_train.groupby(["MicroDeptCode"])["Sugars"].mean()

        micro_mean = pd.DataFrame(micro_mean)
        micro_mean.columns = ["micro_pred"]
        micro_mean["MicroDeptCode"] = micro_mean.index

        micro_test = pd.merge(micro_test, micro_mean,how = "left", on='MicroDeptCode')
        micro_test = micro_test.fillna(0)

        mse = mean_squared_error(y_valid,micro_test['micro_pred'])
        model3.append(mse)

    #
    # print model1
    # print model2
    #
    # print np.mean(model1)
    # print np.mean(model2)
    #
    # print np.std(model1)
    # print np.std(model2)

    result1[sub]=np.mean(model1)
    result2[sub]=np.mean(model2)
    result3[sub]=np.mean(model3)



# importance = model.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
#
# ft.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/feature_importance.csv")
# ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
# plt.gcf().savefig('features_importance.png')

print np.mean(result1.values())
print np.mean(result2.values())
print np.mean(result3.values())

# print result1
# print result1.values()
#
# with open('/Users/xiaofeifei/I/Oxford/Dissertation/some.csv', 'ab') as f:
#     wr = csv.writer(f)
#     wr.writerow(result2.values())

#56.8123535273 ensemble
#59.9441630458 xgb
# 66.8169045656 svr
# 75.4565602039 en
# 70. lasso