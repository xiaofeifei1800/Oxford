
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def ngrams(words, n):
    a = ['_'.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ' '.join(a).lower()

def generateNumber(num):
    mylist = []
    for i in range(num+1):
         mylist.append(str(i))
    return mylist

train_all = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition_clean_data.csv")
train_all = train_all.fillna(0)
main_sub = ["CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45']
#
# main_sub = ["XBISC"]
result1 = {}
result2 = {}
result3 = {}
for sub in main_sub:

    train = train_all.loc[train_all['SubDeptCode'] == sub]

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
        X_train, X_valid, y_train, y_valid = train_test_split(clean_train_reviews, train["Sugars per 100g"], test_size=0.2, random_state=i)

        X_train = pd.DataFrame(X_train)
        X_train.columns = ["unigram"]
        X_train["bigram"] = X_train["unigram"].apply(lambda x: ngrams(x.split(), 2))
        X_train["trigram"] = X_train["unigram"].apply(lambda x: ngrams(x.split(), 3))

        X_valid = pd.DataFrame(X_valid)
        X_valid.columns = ["unigram"]
        X_valid["bigram"] = X_valid["unigram"].apply(lambda x: ngrams(x.split(), 2))
        X_valid["trigram"] = X_valid["unigram"].apply(lambda x: ngrams(x.split(), 3))

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,
                                     max_features=100)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data_features_u = vectorizer.fit_transform(X_train["unigram"])
        test_data_features_u = vectorizer.transform(X_valid["unigram"])

        # train_data_features_b = vectorizer.fit_transform(X_train["bigram"])
        # test_data_features_b = vectorizer.transform(X_valid["bigram"])
        #
        # train_data_features_t = vectorizer.fit_transform(X_train["trigram"])
        # test_data_features_t = vectorizer.transform(X_valid["trigram"])

        # Numpy arrays are easy to work with, so convert the result to an
        # array

        # test_data_features_u = test_data_features_u.toarray()
        # test_data_features_u = pd.DataFrame(test_data_features_u)
        #
        # test_data_features_b = test_data_features_b.toarray()
        # test_data_features_b = pd.DataFrame(test_data_features_b)
        #
        # test_data_features_t = test_data_features_t.toarray()
        # test_data_features_t = pd.DataFrame(test_data_features_t)
        #
        # # test_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/test_1.csv", index=False)
        # train_data_features_u = train_data_features_u.toarray()
        # train_data_features_u = pd.DataFrame(train_data_features_u)
        #
        # train_data_features_b = train_data_features_b.toarray()
        # train_data_features_b = pd.DataFrame(train_data_features_b)
        #
        # train_data_features_t = train_data_features_t.toarray()
        # train_data_features_t = pd.DataFrame(train_data_features_t)
        # train_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_1.csv",index=False)

        # combine alll
        # train_data_features = pd.concat([train_data_features_u,train_data_features_b,train_data_features_t], axis=1)
        # test_data_features = pd.concat([test_data_features_u,test_data_features_b,test_data_features_t], axis=1)


        # feature_len = train_data_features.shape[1]
        # train_data_features.columns = generateNumber(feature_len-1)
        #
        # feature_len = test_data_features.shape[1]
        # test_data_features.columns = generateNumber(feature_len-1)

        train_data_features = train_data_features_u
        test_data_features = test_data_features_u
        train_data_features = train_data_features.toarray()
        test_data_features = test_data_features.toarray()
        train_data_features = pd.DataFrame(train_data_features)
        test_data_features = pd.DataFrame(test_data_features)

        y_train = pd.DataFrame(y_train)

        # y_train.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_y.csv",index=False)

        y_valid = pd.DataFrame(y_valid)
        # y_valid.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/test_y.csv",index=False)


        n_comp = 15

        # PCA
        pca = PCA(n_components=n_comp, random_state=420)
        pca2_results_train = pca.fit_transform(train_data_features)
        # var= pca.explained_variance_ratio_
        # var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        # plt.plot(var1)
        # plt.show()

        pca2_results_test = pca.transform(test_data_features)
        # # tSVD
        # tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
        # tsvd_results_train = tsvd.fit_transform(train_data_features)
        # tsvd_results_test = tsvd.transform(test_data_features)

        # # ICA
        # ica = FastICA(n_components=n_comp, random_state=420)
        # ica2_results_train = ica.fit_transform(train_data_features)
        # ica2_results_test = ica.transform(test_data_features)
        #
        # # GRP
        # grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
        # grp_results_train = grp.fit_transform(train_data_features)
        # grp_results_test = grp.transform(test_data_features)
        #
        # # SRP
        # srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
        # srp_results_train = srp.fit_transform(train_data_features)
        # srp_results_test = srp.transform(test_data_features)

        # x_train = pd.DataFrame()
        # x_test = pd.DataFrame()
        # x_train = x_train.append(pd.DataFrame({'pca_' + str(1): pca2_results_train[:,0]}), ignore_index=True)
        # x_test = x_test.append(pd.DataFrame({'pca_' + str(1): pca2_results_test[:, 0]}), ignore_index=True)
        # Append decomposition components to datasets
        # for i in range(1, n_comp+1):
        #     x_train['ica_' + str(i)] = ica2_results_train[:,i-1]
        #     x_test['ica_' + str(i)] = ica2_results_test[:, i-1]
        #
        #     x_train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
        #     x_test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]
        #
        #     x_train['grp_' + str(i)] = grp_results_train[:,i-1]
        #     x_test['grp_' + str(i)] = grp_results_test[:, i-1]
        #
        #     x_train['srp_' + str(i)] = srp_results_train[:,i-1]
        #     x_test['srp_' + str(i)] = srp_results_test[:, i-1]


        for i in range(1, n_comp+1):
            train_data_features['pca_' + str(i)] = pca2_results_train[:,i-1]
            test_data_features['pca_' + str(i)] = pca2_results_test[:, i-1]


        dtrain = xgb.DMatrix(data =train_data_features, label = y_train)
        dtest = xgb.DMatrix(data =test_data_features)

        params = {
                "max_depth": 15,
                'eta': 0.1,
                'n_trees': 520,
                "eval_metric": "rmse",
                "objective": "reg:linear",
                "nthread" : 6,
                "base_score" : np.mean(y_train["Sugars per 100g"]),
                "early_stopping_rounds": 10,
                'silent': 1
        }

        model = xgb.train(params, dtrain=dtrain)
        pred = model.predict(dtest)

        mse = mean_squared_error(y_valid,pred)
        model2.append(mse)

        mse = mean_squared_error(y_valid, np.repeat(np.mean(y_train["Sugars per 100g"]),y_valid.shape[0]))
        model1.append(mse)

        micro_train = train.ix[list(y_train.index),["MicroDeptCode", "Sugars per 100g"]]
        micro_test = train.ix[list(y_valid.index),["MicroDeptCode", "Sugars per 100g"]]

        micro_mean =  micro_train.groupby(["MicroDeptCode"])["Sugars per 100g"].mean()

        micro_mean = pd.DataFrame(micro_mean)
        micro_mean.columns = ["micro_pred"]
        micro_mean["MicroDeptCode"] = micro_mean.index

        micro_test = pd.merge(micro_test, micro_mean,how = "left", on='MicroDeptCode')
        micro_test = micro_test.fillna(0)

        mse = mean_squared_error(y_valid,micro_test['micro_pred'])
        model3.append(mse)


    # print model1
    # print model2

    # print np.mean(model1)
    # print np.mean(model2)
    # print np.mean(model3)
    #
    # print np.std(model1)
    # print np.std(model2)
    # print np.std(model3)

    result1[sub]=np.mean(model1)
    result2[sub]=np.mean(model2)
    result3[sub]=np.mean(model3)

print np.mean(result1.values())
print np.mean(result2.values())
print np.mean(result3.values())

