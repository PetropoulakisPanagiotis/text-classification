import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get data from file #
text = pd.read_csv('dataSets/train_set.csv', encoding='utf-8', sep="\t")

# Keep only category and content columns #
text = text.drop(['RowNum', 'Id', 'Title'], axis=1)

# Find total categories in data set #
# Group text based on category                               #
# catDict[i]:                                                #
# A list that contains content with category i(i is integer) #
# catDict + categories are "parallel"                        #
categories = list(text.Category.unique())
totalCategories = len(categories)
catDict = dict()

for i in range(totalCategories):

    # Get series with current category #
    tempDF = text.loc[text['Category'] == categories[i]]

    catDict[i] = tempDF["Content"].tolist()

# Generate wordclouds #
for i in range(totalCategories):

    wc = WordCloud(width=1000, height=800, min_font_size=10, max_font_size=450, background_color="#e1b486", max_words=1000)
    wc.generate(catDict[i][0])

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # Save plot in current directory #
    plt.savefig(categories[i])
