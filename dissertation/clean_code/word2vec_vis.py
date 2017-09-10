import re
import gensim
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def name_to_words(raw_name):
    review_text = BeautifulSoup(raw_name).get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000. == 0.:
            reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
        counter = counter + 1.
    return reviewFeatureVecs

def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #     plt.savefig(filename)
    plt.show()

train1 = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train1 = train1.fillna(0)

main_sub = ["CEREAL"]

for sub in main_sub:
    print sub

    train = train1.loc[train1['SubDeptCode'] == sub]
    suger = train["Sugars"]
    suger = suger.reset_index(drop=True)

    if train.shape[0] <= 200:
        continue

    train = train.reset_index(drop=True)

    # Get the number of names based on the dataframe column size
    num_names = train["SkuName"].size

    # clean product names
    clean_names = []

    for i in xrange(0, num_names):
        clean_names.append(name_to_words(train["SkuName"][i]))

    clean_names = []
    for i in xrange(0, num_names):
        clean_names.append(name_to_words(train["SkuName"][i]))

    longest = np.max(map(len, clean_names))

model = gensim.models.Word2Vec(clean_names, size=100, window=longest, min_count=2, workers=4)

trainDataVecs = getAvgFeatureVecs(clean_names, model, 100)

pca = PCA(n_components=2)
pca.fit(trainDataVecs)

embeds = []
labels = []
for i in range(25, 50):
    embeds.append(trainDataVecs[i])
    labels.append(train["SkuName"][i])
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)
