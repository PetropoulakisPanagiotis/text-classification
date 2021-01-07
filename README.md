[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Text Classification
<p align="center">
<img src="https://cdn-images-1.medium.com/max/640/1*ljCBykAJUnvaZcuPYwm4_A.png" width="400" height="250"> <br />
</p>

Classify BBC articles into categories with scikit-learn

## How It Works
There are two data sets. The train_set.csv with 12.267 data points and the test_set.csv with 3.068 data points. The train set contains 5 columns per article. ID, Title, Content, Category(Politics, Film, Football, Business, Technology) and RowNum. Our goal is to find the best classifier for this specific train set and then use it to classify the articles of the test set. 

At first, you can gain an insight into the data set by running the wordcloud.py module to generate one Word Cloud for each category. Then, the next step is to preprocess and convert the content of each article into a vector representation excluding stop-words and using the TFIDF Vectorizer method. After that, there is an additional step where each vector is down-sampled to a lower dimension to reduce the training time of each model and even increase their accuracy, as irrelevant and redundant information may be removed during this step. The best dimension with the best trade-off between accuracy and training time is 100 dimensions(you can take a look into the lsi_plot.png). The next step is to use the 10-fold cross-validation method to train different models with our train set and find the best hyper-parameters(grid_search modules). Then, by running the train_models.py module, you can find which models perform the best to this specific problem(using the train set). After finding the best model and adding some extra preprocessing steps such as Porter Stemming and appending the title to each TFIDF vector, you can run the beat_the_benchmark.py module to predict the categories of the test articles.   

## Requirements
1. Python 2.7
2. Scikit-learn
3. Pandas
4. NLTK
5. matplotlib
6. [wordcloud](https://github.com/amueller/word_cloud)

## Helpful Links: 
1. http://scikit-learn.org/
2. http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
3. http://scikit-learn.org/stable/modules/grid_search.html
4. http://scikit-learn.org/stable/modules/cross_validation.html
5. https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
6. https://elitedatascience.com/overfitting-in-machine-learning
7. http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
8. http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
9. https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
10. https://medium.com/machine-learning-101
11. http://scikit-learn.org/stable/modules/grid_search.html
12. http://scikit-learn.org/stable/modules/cross_validation.html

## Authors
* Petropoulakis Panagiotis petropoulakispanagiotis@gmail.com
* Andreas Charalambous and.charalampous@gmail.com
