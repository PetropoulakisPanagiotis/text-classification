import math
import operator

'''	
	The Class KNNalgorithm is used to implement the k - nearest neighbor algorithm.
	When constructed, it get the k parameter. 
	
	The fit() function is used to keep the train set and the label encoding.
	
	The predict() function gets the unknown data, that will be predicted, in a tfidf 
	vector. Using euclidean distance for n dimension(n = number of words in vector),
	finds the distance from all the texts in train set. It then sorts them in ascending
	order and keeps only the k first of them. Finally, the category of the texts given
	is determined, as it is the same as the category that the most texts in the first k
	have.

'''  	

class KNNClassifier:

    def __init__(self, k):
        self.k = k # parameter K: number of nearest neighbors to be checked

    def fit(self, X, y):
        self.X = X # training set
        self.y = y # label encoder

    def predict(self, X):

        predictions = list() # final predictions to be returned
        
        # Find distance for every text from train set#
        for i in range(X.shape[0]): # for all texts in testset
            dist = list() # create a list with each distance
            
            for j in range(self.X.shape[0]): # check for every text in trainset
                currDist = 0

                for k in range(self.X.shape[1]): # check every word(dimension)
                    currDist += ((X[i, k] - self.X[j, k])**2) # Euclidean Distance
                currDist = math.sqrt(currDist)
                dist.append([currDist, j])

            dist = sorted(dist, key=lambda dist: dist[0]) # sort distances in ascending order
            
            dist = dist[0:self.k] # keep first k


            # find number of occurences for each category in the first k #
            occ = dict()
            for z in range(len(dist)):
                curr = str(self.y[dist[z][1]])
                if curr not in occ:
                    occ[curr] = 1
                else:
                    occ[curr] += 1

            # keep max category repeated #
            predictions.append(int(max(occ.iteritems(), key=operator.itemgetter(1))[0]))

        return predictions
