import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate
from nltk import PorterStemmer
from sklearn.model_selection import StratifiedKFold

pd.set_option('precision', 6)

# Use porter stemmer #
stemmer = PorterStemmer()

# 10-fold #
kf = StratifiedKFold(n_splits=10, random_state=123)

# Read data #
train_data = pd.read_csv('dataSets/train_set.csv', encoding="utf-8", sep="\t")
test_data = pd.read_csv('dataSets/test_set.csv', encoding="utf-8", sep="\t")

# Drop useless columns #
train_data = train_data.drop(['RowNum', 'Id'], axis=1)

y_train = train_data["Category"]
X_train = train_data["Content"]
X_test = test_data["Content"]
X_title = train_data["Title"]
Y_title = test_data["Title"]

# Perform stemming #
lst = []
for i in range(X_train.shape[0]):
    s = X_train.iloc[i]
    x = []
    for t in s.split(" "):
        x.append(stemmer.stem(t))

    lst.append(" ".join(x))

tmp = pd.DataFrame(lst, columns=["Content"])

X_train = tmp["Content"]

# Perform stemming in test set #
lst = []
for i in range(X_test.shape[0]):
    s = X_test.iloc[i]
    x = []
    for t in s.split(" "):
        x.append(stemmer.stem(t))

    lst.append(" ".join(x))

tmp = pd.DataFrame(lst, columns=["Content"])

X_test = tmp["Content"]

# Add labels #
le_train = preprocessing.LabelEncoder()
X_train_le = le_train.fit_transform(y_train)
X_train_cat = le_train.inverse_transform(X_train_le)

# Create matrix of TF-IDF features #
# Use title efficiently            #
tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train + (5 * X_title))
X_test_tfidf = tfidf_vectorizer.transform(X_test + (5 * Y_title))

# Normalize data #
norm = Normalizer()
X_train_tfidf = norm.fit_transform(X_train_tfidf)
X_test_tfidf = norm.transform(X_test_tfidf)

# Classifier #
clf = SVC(C=1, kernel="rbf", gamma=10)

# Use LSA for dimensionality reduction #
svd = TruncatedSVD(n_components=100, random_state=123)

# Perform dimensionality reduction #
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_tfidf = svd.transform(X_test_tfidf)

# Metrics #
scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

# Evaluate my method #
scores = cross_validate(clf, X_train_reduced, X_train_le, scoring=scoring, cv=kf)

# Print results to csv #
Evaluation_metric_df = pd.read_csv('EvaluationMetric_10fold.csv', sep="\t")

Evaluation_metric_df['My Method'] = [str(round(scores['test_accuracy'].mean(), 6)),
                                     str(round(scores['test_precision_macro'].mean(), 6)),
                                     str(round(scores['test_recall_macro'].mean(), 6)),
                                     str(round(scores['test_f1_macro'].mean(), 6))]

# Create csv #
Evaluation_metric_df.to_csv("EvaluationMetric_10fold.csv", encoding='utf-8', index=False, sep="\t")

# Predict test set #

# Train model #
clf.fit(X_train_reduced, X_train_le)

# Predict categories #
y_test = clf.predict(X_test_tfidf)
y_cat = le_train.inverse_transform(y_test)

# Create csv of predicted categories #
cols = ['Id', 'Category']
lst = []

# Lst: list of lists #
# Every single list  #
# contains id and    #
# predicted category #

for i in range(test_data.shape[0]):
    curr_id = test_data.iloc[i]['Id']
    lst.append([curr_id, y_cat[i]])

# Create a dataframe and convert it into csv #
pf = pd.DataFrame(lst, columns=cols)
pf.to_csv("testSet_categories.csv", encoding="utf-8", sep="\t", index=False)
