import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, FastICA
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re


train1 = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train1 = train1.fillna(0)
train = train1.loc[train1['SubDeptCode'] == "CEREAL"]
train = train.reset_index(drop=True)
y_mean = np.mean(train["Sugars"])
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


class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, y)

    def predict(self, X):
        return super(LogExpPipeline, self).predict(X)

#
# Model/pipeline with scaling,pca,svm
# knn
knn_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            KNeighborsRegressor(n_neighbors = 15, metric = 'cityblock')]))
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SVR(kernel='rbf', C=30, epsilon=0.05)]))

# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()

#
# Model/pipeline with scaling,pca,ElasticNet
#
en = ElasticNet(alpha=0.01, l1_ratio=0.9)

#
# XGBoost model
#
xgb_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.921,
                                     objective='reg:linear', n_estimators=1300, base_score=y_mean)


# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestRegressor(n_estimators=500, n_jobs=4, min_samples_split=10,
                                 min_samples_leaf=30, max_depth=3)

# results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))

# ridge
Ridge = Ridge(alpha=37)

# lasso
lasso = LassoLarsCV(normalize=True)

#GBR
gbm = GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18,
min_samples_split=14, subsample=0.7)


#
# Now the training and stacking part.  In previous version i just tried to train each model and
# find the best combination, that lead to a horrible score (Overfit?).  Code below does out-of-fold
# training/predictions and then we combine the final results.
#
# Read here for more explanation (This code was borrowed/adapted) :
#

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y):
        X = np.array(X)
        y = np.array(y)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_valid = X[test_idx]
                y_valid = y[test_idx]

                train2 = train.ix[train_idx]
                test2 = train.ix[test_idx]

                train2 = train2.reset_index(drop=True)
                test2 = test2.reset_index(drop=True)

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

                vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,
                                     ngram_range=(1,2), max_features=100)
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

                for j in range(1, n_comp+1):
                    train_data_features['pca_' + str(j)] = pca2_results_train[:,j-1]
                    test_data_features['pca_' + str(j)] = pca2_results_test[:, j-1]



                train_data_features = pd.concat([train_data_features, train2[[
                "Sugars_mean", "Sugars_min","Sugars_max", "Sugars_var",
                # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
                # "Sugars_var12","Sugars_var13","Sugars_var23",
                # "Sugars_max12","Sugars_max13","Sugars_max23",
                # "Sugars_min12","Sugars_min13","Sugars_min23",
                "Sugars_mean1", "Sugars_min1","Sugars_max1", "Sugars_var1"
                                                                                ,"SkuCode"]]], axis=1)

                test_data_features = pd.concat([test_data_features, test2[[
                "Sugars_mean", "Sugars_min","Sugars_max", "Sugars_var",
                # "Sugars_mean12","Sugars_mean13","Sugars_mean23",
                # "Sugars_var12","Sugars_var13","Sugars_var23",
                # "Sugars_max12","Sugars_max13","Sugars_max23",
                # "Sugars_min12","Sugars_min13","Sugars_min23",
                "Sugars_mean1", "Sugars_min1","Sugars_max1", "Sugars_var1"
                                                                                ,"SkuCode"]]], axis=1)

                train_data_features = train_data_features.fillna(0)
                test_data_features = test_data_features.fillna(0)
                clf.fit(train_data_features, y_train)
                y_pred = clf.predict(test_data_features)[:]

                print ("Model %d fold %d score %f" % (i, j, mean_squared_error(y_valid, y_pred)))


                S_train[test_idx, i] = y_pred.ravel()
            oof_score = mean_squared_error(y, S_train[:, i])
            print 'Final Out-of-Fold Score %f'%oof_score
        return S_train

#knn_pipe,svm_pipe, en,xgb_model, rf_model, Ridge, lasso, gbm
stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker= xgb.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.93,
                                     objective='reg:linear', n_estimators=1300, base_score=y_mean),
                 base_models=(knn_pipe,svm_pipe, en,xgb_model, rf_model, Ridge, lasso, gbm))

S_train = stack.fit_predict(clean_train_reviews, train["Sugars"])

S_train = pd.DataFrame(S_train)

S_train.columns = ["knn", "svm", "en", "xgb", "rf", "Ridge", "lasso", "gbm"]
# S_test.columns = ["knn", "svm", "en", "xgb", "rf", "Ridge", "lasso", "gbm"]
#
S_train.to_csv(path+'stacking_reg_train.csv', index=False)
# S_test.to_csv(path+'stacking_reg_test.csv', index=False)
