import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility


def name_to_words(raw_name):
    review_text = BeautifulSoup(raw_name).get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       counter = counter + 1.
    return reviewFeatureVecs

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

train = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train = train.fillna(0)
train = train.drop(["class"], axis=1)

main_sub = ["CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
            'ICREAM', '335', '338', '390', '396', '45']


train = train.loc[train['SubDeptCode'].isin(main_sub)]
num_reviews = train["SkuName"].size

# clean product names
clean_names = []

for i in xrange(0, num_reviews):
    clean_names.append(name_to_words(train["SkuName"][i]))

clean_names = []
for i in xrange(0, num_reviews):
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_reviews)
    clean_names.append(name_to_words(train["SkuName"][i]))

# create new features
word_len = []
for item in clean_names:
    word_len.append(len(item.split()))

train["word_count"] = word_len

train = train.join(train.groupby('MicroDeptCode')['Sugars'].mean(), on='MicroDeptCode', rsuffix='_mean')
train = train.join(train.groupby('MicroDeptCode')['Sugars'].var(), on='MicroDeptCode', rsuffix='_var')
train = train.join(train.groupby('MicroDeptCode')['Sugars'].min(), on='MicroDeptCode', rsuffix='_min')
train = train.join(train.groupby('MicroDeptCode')['Sugars'].max(), on='MicroDeptCode', rsuffix='_max')

train = train.join(train.groupby('SubDeptCode')['Sugars'].mean(), on='MicroDeptCode', rsuffix='_mean_s')
train = train.join(train.groupby('SubDeptCode')['Sugars'].var(), on='MicroDeptCode', rsuffix='_var_s')
train = train.join(train.groupby('SubDeptCode')['Sugars'].min(), on='MicroDeptCode', rsuffix='_min_s')
train = train.join(train.groupby('SubDeptCode')['Sugars'].max(), on='MicroDeptCode', rsuffix='_max_s')

train = train.join(train.groupby(['code1'])['Sugars'].mean(), on=['code1'], rsuffix='_mean1')
train = train.join(train.groupby('code1')['Sugars'].var(), on='code1', rsuffix='_var1')
train = train.join(train.groupby('code1')['Sugars'].min(), on='code1', rsuffix='_min1')
train = train.join(train.groupby('code1')['Sugars'].max(), on='code1', rsuffix='_max1')

# transfer names to tf-ide and word2vec
vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                             ngram_range=(1, 1), max_features=100)

train_data_features = vectorizer.fit_transform(clean_names)
train_data_features = train_data_features.toarray()
train_data_features = pd.DataFrame(train_data_features)

model = Word2Vec(clean_names, workers=6, size=100, min_count = 2, window = 1, seed=1)

DataVecs = getAvgFeatureVecs(getCleanReviews(clean_names), model, 100)
DataVecs = pd.DataFrame(DataVecs)

n_comp = 15

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train_data_features)
pca_word2vec = pca.fit_transform(DataVecs)

for i in range(1, n_comp + 1):
    train_data_features['pca_' + str(i)] = pca2_results_train[:, i - 1]

for i in range(1, n_comp + 1):
    train_data_features['pca_word2vec_' + str(i)] = pca_word2vec[:, i - 1]

train_data_features = pd.concat([train_data_features, DataVecs, train[[
    "Sugars_mean", "Sugars_min", "Sugars_max", "Sugars_var",
    "Sugars_mean_s", "Sugars_min_s", "Sugars_max_s", "Sugars_var_s",
    "code1", "code2", "code3",
    "Sugars_mean1", "Sugars_min1", "Sugars_max1", "Sugars_var1",
    "SkuCode", 'Sugars',"OwnBrand"]]], axis=1)


print train_data_features.columns
train_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_feature.csv",index=False)






























