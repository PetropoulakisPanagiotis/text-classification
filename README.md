[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Text classification
<p align="center">
<img src="https://cdn-images-1.medium.com/max/640/1*ljCBykAJUnvaZcuPYwm4_A.png" width="400" height="250"> <br />
</p>

Classify BBC articles into categories with scikit-learn

## How It Works
There are two data sets. train_set.csv with 12.267 data points and test_set.csv with 3.068 data points. Train set contains 5 columns per article. ID, Title, Content, Category(Politics or Film or Football or Business or Technology) and RowNum. Our goal is to find the best classifier for this specific train set, and then use it to classify the articles of the test set. 

At first, you can gain an insight into the data set by running the wordcloud.py module to generate one Wordcloud for each category. Then, the next step is to prerpocce and convert the content of each article to a vector representation excluding stop-words and using the TFIDF Vectorizer method. After that, there is an additional step where each vector is downsampled to a lower dimension so as to reduce the execution time that each model needs to be trained and even increase the accurasy, as irelavant and redundant information can be removed. The best dimension with the best trade-off between accuracy and axecution time is 100 dimension(you can take a look into the lsi_plot.png). The next step is to use the 10 fold cross validation method to train different models with our train set to find the best hyper-parameters(grid_search modules) for our problem. Then, 

Read the following list to comprehend this repository: 
<br />
1. At first run the  
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
