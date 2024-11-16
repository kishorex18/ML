import numpy as np
import pandas as pd
from itertools import combinations

data=pd.read_csv("Student_Performance.csv")
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


def coefficients(X, Y, learning_rate=0.01, iterations=1000):
    # Add a bias term (intercept) to the features
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize parameters (theta) to zeros
    theta = np.zeros(X_b.shape[1])
    
    # Gradient Descent loop
    m = len(Y)
    for _ in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - Y
        
        # Compute gradients for each parameter
        gradient = (1/m) * X_b.T.dot(errors)
        
        # Update the parameters (theta)
        theta = theta - learning_rate * gradient
    
    return theta

def predict(X,theata):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return X_b.dot(theata)
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def forward_stepwise_selection(X_train,Y_train,X_test,Y_test):
    remaining_features=list(X_train.columns)
    selected_features=[]
    best_mse=float("inf")
    best_theata=None

    while remaining_features:
        mse_with_current_features=[]
        for feature in remaining_features:
            selected_features_candidate=selected_features+[feature]
            X_train_subset=X_train[selected_features_candidate]
            X_test_subset=X_test[selected_features_candidate]
            theata=coefficients(X_train_subset,Y_train)
            Y_pred=predict(X_test_subset,theata)
            current_mse=mse(Y_pred,Y_test)
            mse_with_current_features.append((current_mse,feature,theata))

        best_current_mse,best_features,best_current_theata=min(mse_with_current_features,key=lambda x:x[0])
        if best_current_mse<best_mse:
            best_mse=best_current_mse
            selected_features.append(best_features)
            best_theata=best_current_theata
            remaining_features.remove(best_features)
        else:
            break
    
    return best_mse,selected_features,best_theata
best_mse,selected_features,best_theata=forward_stepwise_selection(X_train,Y_train,X_test,Y_test)
print(best_mse)
print(selected_features)
print(best_theata)
