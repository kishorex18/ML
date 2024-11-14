import numpy as np
import pandas as pd

data=pd.read_csv('multiple_linear_regression_dataset.csv')

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

X=np.c_[np.ones(X.shape[0]),X]

def train_test_split(X,Y,test_size=0.2):
    indices=np.random.permutation(X.shape[0])
    test_size=int(X.shape[0]*test_size)
    test_indices=indices[:test_size]
    train_indices=indices[test_size:]
    X_train,Y_train=X[train_indices],Y[train_indices]
    X_test,Y_test=X[test_indices],Y[test_indices]
    return X_train,Y_train,X_test,Y_test
X_train,Y_train,X_test,Y_test=train_test_split(X,Y)

def normal_equation(X,Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y
def train_and_evaluate(X_train,Y_train,X_test,Y_test):
    beta=normal_equation(X_train,Y_train)
    Y_pred_test=X_test @ beta

    rmse_test=np.sqrt(np.mean((Y_test-Y_pred_test)**2))
    return beta,rmse_test

beta,rmse_test=train_and_evaluate(X_train,Y_train,X_test,Y_test)
print("RMSE",rmse_test)
print("Beta",beta)