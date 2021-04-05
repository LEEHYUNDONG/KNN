import numpy as np
from collections import Counter

#knn class
class KNN:
    #initialization 
    def __init__(self, X, y, y_name):
        print('__init__')
        self.X = X
        self.y = y
        self.y_name = y_name
        self.X_train = np.array([])
        self.y_train = np.array([])

    #split data to 14: 1 every 15th data will be used for test_data    
    def split_data(self, X, y):
        #Set the data 
        X_train = np.array([])
        X_test  = np.array([])
        y_train = np.array([])
        y_test = np.array([])        
        
        for i in range(0, len(X)):
            if (i+1)%  15 != 0:
                X_train = np.append(X[i], X_train)
                y_train = np.append(y[i], y_train)
            else:
                #print(i)
                X_test = np.append(X[i], X_test)
                y_test = np.append(y[i], y_test)
        
        # if we append data to new np array it append backward so have to reverse(mirror) data for sure
        X_train = X_train[::-1]
        X_test = X_test[::-1]
        y_train = y_train[::-1]
        y_test = y_test[::-1]
        
        #reshape the data
        X_train = X_train.reshape((140, 4))
        X_test = X_test.reshape((10, 4))   
        y_train = y_train.reshape((140, 1))
        y_test = y_test.reshape((10, 1))
        self.X_train = X_train
        self.y_train = y_train

        return X_train, X_test, y_train, y_test

    # To calculate euclidean distance and return results
    def euclideanDistance(self, x1, x2):
        return np.sqrt(((x2-x1)**2).sum())
    
    #get distance matrix
    def getDistance(self, Points, plot):
        dist = np.zeros(len(self.X_train)) 
        
        for i in range(len(self.X_train)):
            #dist[i] = self.euclideanDistance(Points[plot], self.X_train[i])
            dist[i] = self.euclideanDistance(self.X_train[i], Points[plot])

        return dist

    #majority vote for k
    def majorityVote(self, targets, dist, k):  #k is the number of neighbours
        dist_index = np.argsort(dist)[0:k]
        res = np.zeros(k)
        
        for i in range(k):
            res[i] = targets[dist_index[i]]

        #val, cnts = np.unique(res, return_counts=True)
        cnts = Counter(res)
        #print(cnts)
        return cnts.most_common(1)[0][0]
    
    #weighted majority vote
    def weightedMajorityVote(self, targets, dist, k):

        for i in range(len(dist)):
            dist[i] = (1/1+dist[i])*dist[i]

        dist_index = np.argsort(dist)[0:k] #sort data by index because we need the nearby data's indices
        res = np.zeros(k) #initialize res np array to zero amount of size k
        for i in range(k):
            res[i] = targets[dist_index[i]] 
        cnts = Counter(res)
        
        return cnts.most_common(1)[0][0] #return the most common data
    
    def knn(self, className, targets, Points, plot, k):
        dist = self.getDistance(Points, plot)
        #return className[int(self.majorityVote(targets, dist, k))]
        return className[int(self.weightedMajorityVote(targets, dist, k))]
    
    #Output data
    def printResult(self, Points, targets, k):
        for i in range(len(targets)):
            #have to pass y_train data for target cause we get each of the plot's distances from training data 
            print("Test Data Index: ", i, "Computed class: ", self.knn(self.y_name, self.y_train, Points, i, k), 
                ", True class : ", self.y_name[int(targets[i])])




