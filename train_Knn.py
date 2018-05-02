import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from knn import KNNClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

pd.set_option('precision', 6)

# Read data #
train_data = pd.read_csv('dataSets/train_set.csv', encoding='utf-8', sep="\t")

# Drop useless columns #
train_data = train_data.drop(['RowNum', 'Id', 'Title'], axis=1)

# 10-fold #
kf = StratifiedKFold(n_splits=10, random_state=123)

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


# Keep results #
prec, rec, f1, accu = ([] for i in range(4))

# Use 10-fold and find metrics #
for train, test in kf.split(X_train_reduced, X_train_le):
    X_train = X_train_reduced[train]
    y_train = X_train_le[train]

    X_test = X_train_reduced[test]
    y_test = X_train_le[test]

    clf_KNN = KNNClassifier(100)

    # Train model #
    clf_KNN.fit(X_train, y_train)

    # Predict categories #
    y_pred = clf_KNN.predict(X_test)

    # Save scores #
    prec.append(precision_score(y_test, y_pred, average='macro'))
    rec.append(recall_score(y_test, y_pred, average='macro'))
    f1.append(f1_score(y_test, y_pred, average='macro'))
    accu.append(accuracy_score(y_test, y_pred))


# Print results to csv #
Evaluation_metric_df = pd.read_csv('EvaluationMetric_10fold.csv', sep="\t")

Evaluation_metric_df['KNN'] = [round(sum(accu) / float(len(accu)), 6),
                               round(sum(prec) / float(len(prec)), 6),
                               round(sum(rec) / float(len(rec)), 6),
                               round(sum(f1) / float(len(f1)), 6)]

# Create csv #
Evaluation_metric_df.to_csv("EvaluationMetric_10fold.csv", encoding='utf-8', index=False, sep="\t")
