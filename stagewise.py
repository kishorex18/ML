import pandas as pd
import numpy as np

data=pd.read_csv("Student_Performance.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

def split(X,Y):
    train_size=0.8
    split_index=int(len(Y)*train_size)
    X_train,X_test=X[:split_index],X[split_index:]
    Y_train,Y_test=Y[:split_index],Y[split_index:]
    return X_train,X_test,Y_train,Y_test

def mse(Y_pred,Y):
    return np.mean((Y_pred-Y)**2)

def predictions(X,beta):
     n_samples=X.shape[0]
     X_bias=np.c_[np.ones((n_samples,1)),X]
     return X_bias @ beta

def stepwise(X,Y,step_size,max_steps):
    n,p=X.shape
    X_bias=np.c_[np.ones((n,1)),X]
    beta=np.zeros(p+1)
    predictions=np.mean(Y)*np.ones_like(Y)

    for step in range(max_steps):
        residual=Y-predictions
        correlations=np.dot(X.T,residual)
        best_feature=np.argmax(np.abs(correlations))
        beta[best_feature]+=step_size*np.sign(correlations[best_feature])
        predictions=np.dot(X_bias,beta)
    return beta
X_train,X_test,Y_train,Y_test=split(X,Y)
beta=stepwise(X_train,Y_train,0.01,1000)
Y_pred=predictions(X_test,beta)
current_mse=mse(Y_pred,Y_test)
print("beta:",beta)
print("mse:",current_mse)
