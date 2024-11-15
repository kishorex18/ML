import numpy as np
import pandas as pd
from itertools import combinations

data=pd.read_excel("Real estate valuation data set.xlsx")
X=data.iloc[:,:-1]
Y=data.iloc[:,-1].values

def train_test_split(X,Y,test_size=0.2):
    indices=np.random.permutation(X.index)
    test_size=int(X.shape[0]*test_size)
    test_indices=indices[:test_size]
    train_indices=indices[test_size:]
    X_train,Y_train=X.iloc[train_indices],Y[train_indices]
    X_test,Y_test=X.iloc[test_indices],Y[test_indices]
    return X_train,Y_train,X_test,Y_test
X_train,Y_train,X_test,Y_test=train_test_split(X,Y,0.2)

def coeffecients(X,Y):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    theata_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return theata_best
def predict(X,theata):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return X_b.dot(theata)
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def best(X_train,Y_train,X_test,Y_test):
     best_mse=float('inf')
     best_subset=None
     best_theata=None

     for k in range(1,len(X_train.columns)+1):
         for subset in combinations(X_train.columns,k):
             X_subset_train=X_train[list(subset)]
             X_subset_test=X_test[list(subset)]
             theata=coeffecients(X_subset_train,Y_train)
             Y_pred=predict(X_subset_test,theata)
             current_mse=mse(Y_test,Y_pred)
             if current_mse<best_mse:
                 best_mse=current_mse
                 best_subset=subset
                 best_theata=theata
     return best_mse,best_theata,best_subset  
best_mse,best_theata,best_subset=best(X_train,Y_train,X_test,Y_test)
#print("best features:",type(best_subset))
#print("best coefficients:",best_theata)
#print("Best mse:",best_mse)
print(best_subset)
print(best_theata)

