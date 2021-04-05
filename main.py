import KNN  #import knn class 
from sklearn.datasets import load_iris

#main
#get the Data from iris API
iris = load_iris()
type(iris)
X = iris.data
y = iris.target
y_name = iris.target_names
y = iris.target

# SET 'A' OBJECT FOR TEST
a = KNN.KNN(X, y, y_name)

#SPLIT THE TEST, TRAIN DATA EVERY 15TH DATA WILL USED FOR TEST DATA REST OF THE DATA WILL BE USED FOR TRAINUNG 
#SETS
X_train, X_test, y_train, y_test= a.split_data(X, y)


#print computed classes and true class 
print('-------------------------------------------------------------------------')
print("k == 3")
a.printResult(X_test, y_test, 3)

print('-------------------------------------------------------------------------')
print("k == 5")
a.printResult(X_test, y_test, 5)


print('-------------------------------------------------------------------------')
print("k == 10")
a.printResult(X_test, y_test, 10)


