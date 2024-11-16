import numpy as np
import pandas as pd

data=pd.read_csv("Student_Performance.csv")

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

def ridge_regress(X,Y,alpha):
    n_samples,n_features=X.shape
    X_with_bias=np.c_[np.ones((n_samples,1)),X]
    I=np.eye(n_features+1)
    I[0,0]=0
    theta=np.linalg.inv(X_with_bias.T @ X_with_bias + alpha) @ X_with_bias.T @ Y
    return theta

def predict(X,theta):
    n_samples=X.shape[0]
    X_with_bias=np.c_[np.ones((n_samples,1)),X]
    return X_with_bias @ theta
def mse(Y_predict,Y):
    return np.mean((Y_predict-Y)**2)
def data_split(X,Y):
     split_ratio=0.8
     split_index=int(len(Y)*split_ratio)
     X_train,X_test=X[:split_index],X[split_index:]
     Y_train,Y_test=Y[:split_index],Y[split_index:]
     return X_train,X_test,Y_train,Y_test
X_train,X_test,Y_train,Y_test=data_split(X,Y)
theta=ridge_regress(X_train,Y_train,1.0)
Y_predict=predict(X_test,theta)
current_mse=mse(Y_predict,Y_test)

print("theta:",theta)
print("current_mse:",current_mse)


       
