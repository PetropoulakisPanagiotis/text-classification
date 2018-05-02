import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Read data #
train_data = pd.read_csv('dataSets/train_set.csv', encoding='utf-8', sep="\t")

# Drop useless columns #
train_data = train_data.drop(['RowNum', 'Id', 'Title'], axis=1)

y_train = train_data["Category"]
X_train = train_data["Content"]

# Add labels #
le = preprocessing.LabelEncoder()
X_train_le = le.fit_transform(y_train)
X_train_cat = le.inverse_transform(X_train_le)

# Create matrix of TF-IDF features #
tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Use LSA for dimensionality reduction #
svd = TruncatedSVD(n_components=100, random_state=123)

# Perform dimensionality reduction #
X_train_reduced = svd.fit_transform(X_train_tfidf)

# 10-fold #
kf = StratifiedKFold(n_splits=10, random_state=123)

# Classifier #
clf = RandomForestClassifier()

# RF                                          #
# Note: Hyperparameters will be selected to   #
# be the best based also on time to train the #
# model                                       #
# Best hyperparameters                        #
# n_estimators=100                            #
# criterion=entropy                           #
# max_features=auto                           #
# bootstrap=False                             #
# warm_start=True                             #

# Tune hyperparameters #
parameters = {
            "n_estimators": [10, 30, 100],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt"],
            "bootstrap": [False],
            "warm_start": [True],
            "random_state": [123]
            }

# Notes                          #
# n_estimators: number of trees  #
# criterion: how to split tree   #
# gini: the probability of a     #
# random sample being classified #
# correctly if we randomly pick  #
# a label according to the       #
# distribution in a branch       #
# entropy: measurement of        #
# information  -> may be slower  #
# max_features: number of        #
# features to consider when      #
# looking for the best split     #
# min_samples_split: The minimum #
# number of samples required to  #
# split an internal node:        #
# bootstrap: bootstrap samples   #
# warm_start: reuse trees        #

# Use grid search with 10-fold cross validation #
gs_clf = GridSearchCV(clf, parameters, cv=kf)
gs_clf = gs_clf.fit(X_train_reduced, X_train_le)

# Print results #
print("Random forest best parameters: ")
print(gs_clf.best_params_)
