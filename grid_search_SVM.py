import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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
clf = SVC()

# SVM                                         #
# Note: Hyperparameters will be selected to   #
# be the best based also on time to train the #
# model                                       #
# Best hyperparameters                        #
# Kernel: linear                              #
# C: 5                                        #
# Gamma: auto                                 #

# Tune hyperparameters #

parameters = {
            "C": [1.0, 5, 0.05],
            "kernel": ["rbf", "linear"],
            "gamma": ["auto", 50, 500],
            "random_state": [123]
            }

# Notes:                                                            #
# C: avoid misclassifying each training example                     #
# Kernel: seperation algorithm                                      #
# Gamma: how far the influence of a single training example reaches #

gs_clf = GridSearchCV(clf, parameters, cv=kf)
gs_clf = gs_clf.fit(X_train_reduced, X_train_le)

print("Support Vector Machines best parameters: ")
print(gs_clf.best_params_)
