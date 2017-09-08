import re
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances, linear_kernel
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLarsCV
from matplotlib import pylab as plt
import operator
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


train1 = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train1 = train1.fillna(0)
train1 = train1.drop(["class"], axis=1)

main_sub = ["CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
            'ICREAM', '335', '338', '390', '396', '45']

for sub in main_sub:
    print sub

    train = train1

    train = train.reset_index(drop=True)


    def review_to_words(raw_review):
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
        return (" ".join(meaningful_words))


    # Get the number of reviews based on the dataframe column size
    num_reviews = train["SkuName"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    for i in xrange(0, num_reviews):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append(review_to_words(train["SkuName"][i]))

    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if ((i + 1) % 1000 == 0):
            print "Review %d of %d\n" % (i + 1, num_reviews)
        clean_train_reviews.append(review_to_words(train["SkuName"][i]))

    print "Creating the bag of words...\n"

    X_train, X_valid, y_train, y_valid = train_test_split(clean_train_reviews, train["Sugars"], test_size=0.2,
                                                          random_state=1)
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
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 ngram_range=(1, 2), max_features=100)
    # one way
    train2 = train2.join(train2.groupby(['code1'])['Sugars'].mean(), on=['code1'], rsuffix='_mean1')
    test2 = test2.join(train2.groupby(['code1'])['Sugars'].mean(), on=['code1'], rsuffix='_mean1')

    train2 = train2.join(train2.groupby('code1')['Sugars'].var(), on='code1', rsuffix='_var1')
    test2 = test2.join(train2.groupby('code1')['Sugars'].var(), on='code1', rsuffix='_var1')

    train2 = train2.join(train2.groupby('code1')['Sugars'].min(), on='code1', rsuffix='_min1')
    test2 = test2.join(train2.groupby('code1')['Sugars'].min(), on='code1', rsuffix='_min1')

    train2 = train2.join(train2.groupby('code1')['Sugars'].max(), on='code1', rsuffix='_max1')
    test2 = test2.join(train2.groupby('code1')['Sugars'].max(), on='code1', rsuffix='_max1')

    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 ngram_range=(1, 1), max_features=100)


    train_data_features = vectorizer.fit_transform(X_train)

    test_data_features = vectorizer.transform(X_valid)
    test_data_features = test_data_features.toarray()
    test_data_features = pd.DataFrame(test_data_features)


    train_data_features = train_data_features.toarray()
    train_data_features = pd.DataFrame(train_data_features)

    y_train = pd.DataFrame(y_train)

    y_valid = pd.DataFrame(y_valid)


    n_comp = 15

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train_data_features)
    pca2_results_test = pca.transform(test_data_features)


    for i in range(1, n_comp + 1):
        train_data_features['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test_data_features['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train_data_features = pd.concat([train_data_features, train2[[
        "Sugars_mean", "Sugars_min", "Sugars_max", "Sugars_var",
        # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
        # "Sugars_var12","Sugars_var13","Sugars_var23",
        # "Sugars_max12","Sugars_max13","Sugars_max23",
        # "Sugars_min12","Sugars_min13","Sugars_min23",
        "Sugars_mean1", "Sugars_min1", "Sugars_max1", "Sugars_var1", "word_count",
        "SkuCode"]]], axis=1)

    test_data_features = pd.concat([test_data_features, test2[[
        "Sugars_mean", "Sugars_min", "Sugars_max", "Sugars_var",
        # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
        # "Sugars_var12","Sugars_var13","Sugars_var23",
        # "Sugars_max12","Sugars_max13","Sugars_max23",
        # "Sugars_min12","Sugars_min13","Sugars_min23",
        "Sugars_mean1", "Sugars_min1", "Sugars_max1", "Sugars_var1", "word_count",
        "SkuCode"]]], axis=1)

    feature_names = list(train_data_features.columns.values)

    create_feature_map(feature_names)

    dtrain = xgb.DMatrix(data=train_data_features, label=y_train)
    dtest = xgb.DMatrix(data=test_data_features)

    params = {
        "max_depth": 15,
        'eta': 0.1,
        'n_trees': 1000,
        "eval_metric": "rmse",
        "objective": "reg:linear",
        "nthread": 6,
        "base_score": np.mean(y_train["Sugars"]),
        "early_stopping_rounds": 10,
        'silent': 1
    }

    model = xgb.train(params, dtrain=dtrain)
    pred = model.predict(dtest)

    # S_train = np.zeros((test_data_features.shape[0], 4))
    #
    # S_train[np.arange(test_data_features.shape[0]), 0] = pred.ravel()
    #
    # ######## svm ###########
    # svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
    #                                             SVR(kernel='rbf', C=30, epsilon=0.05)]))
    # train_data_features = train_data_features.fillna(0)
    # test_data_features = test_data_features.fillna(0)
    # svm_pipe.fit(train_data_features, y_train)
    # pred = svm_pipe.predict(test_data_features)[:]
    #
    # S_train[np.arange(test_data_features.shape[0]), 1] = pred.ravel()
    #
    # ############### en #############
    #
    # en = ElasticNet(alpha=0.01, l1_ratio=0.9)
    # en.fit(train_data_features, y_train)
    # pred = en.predict(test_data_features)[:]
    #
    # S_train[np.arange(test_data_features.shape[0]), 2] = pred.ravel()
    #
    # ############ lasso ############
    # lasso = LassoLarsCV(normalize=True)
    # lasso.fit(train_data_features, y_train)
    # pred = lasso.predict(test_data_features)[:]



importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')


