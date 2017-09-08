import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import re
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# train_orders = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/order_products__train.csv", nrows = 100)
# products = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/products.csv").set_index('product_id')
#
# train_orders["product_id"] = train_orders["product_id"].astype(str)
# train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
#
# sentences = train_products
# longest = np.max(sentences.apply(len))
# sentences = sentences.values

train1 = pd.read_csv("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
train1 = train1.fillna(0)

suger = train1["Sugars"]
suger = suger.reset_index(drop=True)



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
num_reviews = train1["SkuName"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in xrange(0, num_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append(review_to_words(train1["SkuName"][i]))

print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []
for i in xrange(0, num_reviews):
    # If the index is evenly divisible by 1000, print a message
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_reviews)
    clean_train_reviews.append(review_to_words(train1["SkuName"][i]))

print "Creating the bag of words...\n"

vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                             ngram_range=(1, 1), max_features=100)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
train_data_features = pd.DataFrame(train_data_features)
n_comp = 4

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train_data_features)

for i in range(1, n_comp + 1):
    train_data_features['pca_' + str(i)] = pca2_results_train[:, i - 1]

pca2_results_train = pd.DataFrame(pca2_results_train)
train_data_features = pd.concat([pca2_results_train, train1[["Sugars", "SkuCode"]]], axis=1)

train_data_features.to_csv("/Users/xiaofeifei/I/Oxford/Dissertation/k_mean_data.csv", header=True, index=False)
