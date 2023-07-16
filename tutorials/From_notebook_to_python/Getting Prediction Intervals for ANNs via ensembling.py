#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Getting Prediction Intervals for ANNs via ensembling
# 
# </font>
# </div>

# # Import modules
# <hr style="border:2px solid black"> </hr>

# In[18]:


import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# # Load the dataset
# <hr style="border:2px solid black"> </hr>

# In[4]:


path = "../DATASETS/housing_1.csv"
dataframe = read_csv(path, header=None)
values = dataframe.values
# split into input and output values
X, y = values[:, :-1], values[:,-1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)
# scale input data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Modelling & fitting
# <hr style="border:2px solid black"> </hr>

# In[10]:


# Define a function because we'll call it several times
def fit_model(X_train, y_train):
    # define neural network model
    features = X_train.shape[1]
    model = Sequential()
    model.add(Dense(20, kernel_initializer='he_normal',
              activation='relu', input_dim=features))
    model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1))
    # compile the model and specify loss and optimizer
    opt = Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999)
    model.compile(optimizer=opt, loss='mse')
    # fit the model on the training dataset
    model.fit(X_train, y_train, verbose=0, epochs=300, batch_size=16)
    return model


# # Getting PIs via ensembling
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - There are many ways to calculate prediction intervals for ANNs and more importantly, there is no standard way.
# - In here we'll shwo a very simple but limited approach: essentially **quick and dirty**.
# - We'll fit anything between 10 to 30 models. The distribution of the point predictions from ensemble members is then used to calculate both a point prediction and a prediction interval.
# - Once all the predictions are computed, we get the mean and use as a point prediction, then to get the symmetric PI we'll comute the +/- 1.96 standard deviation.
# - 1.96 represents the 95% percentile.
# - Because of its symmetric nature it is generally referred to as **Gaussian PI**.
# 
# </font>
# </div>

# In[11]:


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(X_train, y_train)
        # evaluate model on the test set
        yhat = model.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        ensemble.append(model)
    return ensemble


# In[20]:


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X):
    # make predictions
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = np.asarray(yhat)
    # calculate 95% gaussian prediction interval
    interval = 1.96 * yhat.std()
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


# # Run the method
# <hr style="border:2px solid black"> </hr>

# In[22]:


# fit ensemble
n_members = 30
ensemble = fit_ensemble(n_members, X_train, X_test, y_train, y_test)
# make predictions with prediction interval
newX = np.asarray([X_test[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX)
print('Point prediction: %.3f' % mean)
print('95%% prediction interval: [%.3f, %.3f]' % (lower, upper))
print('True value: %.3f' % y_test[0])


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://machinelearningmastery.com/prediction-intervals-for-deep-learning-neural-networks/
# - [High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach](https://arxiv.org/abs/1802.07167)
# - [Practical Confidence and Prediction Intervals](https://papers.nips.cc/paper/1996/hash/7940ab47468396569a906f75ff3f20ef-Abstract.html)
# 
# </font>
# </div>

# In[ ]:




