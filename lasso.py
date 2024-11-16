import numpy as np
import pandas as pd

data=pd.read_csv("Student_Performance.csv")

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

def standardize(X):
    """
    Standardizes the feature matrix X (mean=0, std=1 for each feature).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

def lasso_regress(X,Y,alpha,learning_rate,max_iterations=10000,threshold=1e-4):
    n_samples,n_features=X.shape
    X_with_bias=np.c_[np.ones((n_samples,1)),X]
    weights=np.zeros(n_features+1)

    for it in range(max_iterations):
        predictions=X_with_bias @ weights
        errors=predictions-Y
        gradient=(1/n_samples)*(X_with_bias.T @ errors)

        for j in range(1,len(weights)):
            if weights[j]>0:
                gradient[j]+=alpha
            elif weights[j]<0:
                gradient[j]-=alpha
            else:
                gradient[j]+=0
        weights-=learning_rate*gradient

        if np.linalg.norm(gradient) < threshold:
            print(f"Converged at iteration {it}")
            break
    return weights
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
X, X_mean, X_std = standardize(X)
X_train,X_test,Y_train,Y_test=data_split(X,Y)
theta=lasso_regress(X_train,Y_train,1.0,0.001)
Y_predict=predict(X_test,theta)
current_mse=mse(Y_predict,Y_test)

print("theta:",theta)
print("current_mse:",current_mse)


       
