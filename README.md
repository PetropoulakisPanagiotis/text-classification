[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Text classification
<p align="center">
<img src="https://cdn-images-1.medium.com/max/640/1*ljCBykAJUnvaZcuPYwm4_A.png" width="400" height="250"> <br />
</p>

Classify BBC articlesinto categories

## How It Works(under construction**)
There are two data sets: train_set.csv (12267 data points), test_set.csv (3068 data points). <br />
Train set has 5 fields: Id, Title, Content, Category(Politics/Film/Football/Business/Technology), RowNum.
Our goal is to find the best classifier for our train set and predict the categories of the articles of the test set. 
1. At first run the wordcloud.py module. Wordcloud generates one file per category. 
2. Before evaluating the classifiers we should find a better dimension for our articles(using lsi with constant classifier). In this way we could predict new categories faster, because we have less features after lsi. Lsi plot visualizes how the number of features affect accuracy.
3. The best dimension is settled, so we can tune the hyper-parameters of the classifiers with grid search and cross-validation(avoid over-fitting) -> grind_search modules
4. After finding the best hyper-parameters for classifiers we can select the best classifier -> train_models.py 
5. In the end we can use the title as information and the porter stemmer to increase our predictions -> beat_the_benchmark.py <br />
Note: Some modules produce csv files with some results.

## Requirements
1. Python 2.7
2. Scikit-learn
3. Pandas
4. NLTK
5. matplotlib
6. wordcloud(https://github.com/amueller/word_cloud)

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
