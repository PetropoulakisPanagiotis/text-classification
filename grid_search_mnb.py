import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

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

# 10-fold #
kf = StratifiedKFold(n_splits=10, random_state=123)

# Classifier #
clf = MultinomialNB()

# MNB                                          #
# Note: Hyperparameters will be selected to    #
# be the best, based also on time to train the #
# model                                        #
# Best hyperparameters                         #
# alpha=0.02                                   #
# fit_prior=True                               #

# Tune hyperparameters #
parameters = {
            "alpha": [50, 15, 10, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01,  0.001],
            "fit_prior": [True, False],
            }

# Use grid search with 10-fold cross validation #
gs_clf = GridSearchCV(clf, parameters, cv=kf)
gs_clf = gs_clf.fit(X_train_tfidf, X_train_le)

# Print results #
print("MultinomialNB best parameters: ")
print(gs_clf.best_params_)
