
Missing Data Imputation by Textual Data for Food Product
-----------------------

This is my master dissertation. The goal of this dissertaion is using machine learning and data mining techniques to improve the ecoVerias current imputation method. EcoVerias method is that build a 3 level heriarchy structure (department, sub_department, micro-department) of the food product, and use the mean value of micro-department to replace the missing values. Because the products in micro-department are similar to each other.

The challenges that we faced in this dissertaion are:
1. How to convert this problem to a machine learning problem.
2. The limitaion of the less number of features (we only can use six variables which are department, sub_department, micro-department, product names, product ID, and own brand.)

We define this is a regression problem and evaluate by MSE. We use the non-missing data as training set, and the miss data as test set, and encode heriarchy structure as categorical variables.

Based on our exploratory analysis, we find the product names have a potential affect on missing value. We decide to use this feature into our model by transforming the words to vectors and also use it as another way to group similar products. Once we have the word vectors, we can expand the feature by NLP.

The best model we have obtained during the dissertaion was ensemble the top 4 best models we have with MSE score 46.65, which improve 44.49% compared to ecoVerias method(MSE score 84.04).  

----------------------
### FlowChart
<img src="./Doc/pipline.png" alt="FlowChart" align="center" width="700px"/>

### Download the data

* Because of the confidentiality agreement, I cannot provide the data, but I suggest to crawl some data online to run the scripts

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 2.7.
    * You may want to use a virtual environment for this.

Usage
-----------------------

* Switch to `code/feature` directory using `cd code/feature`
* Run all R and Python files except `combine_all_features.R` to generate new features. The features files are all parallel, so they can be run in any order.
* Run `combine_all_features.R` after run all other files in `code/feature`.(it is not necessary to run this file, combining features also can be customed in `model/lgbm.py`)
* Switch to `model` dirctory
* Run `lgbm.py`.
    * This will dirctly train the lgbm model without CV, and predict the probability of each products
* Run `F1 score max.py`.
    * This will select the reordered products for each customer by maximizing F1 score.(This script is mainly contributed by faron)

