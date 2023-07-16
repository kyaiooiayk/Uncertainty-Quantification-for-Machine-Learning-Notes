#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Create-a-test-function-&amp;-dataset" data-toc-modified-id="Create-a-test-function-&amp;-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create a test function &amp; dataset</a></span></li><li><span><a href="#An-alternative-(with-assumptions)" data-toc-modified-id="An-alternative-(with-assumptions)-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>An alternative (with assumptions)</a></span><ul class="toc-item"><li><span><a href="#Option-#1---Training-a-machine-learning-model-to-predict-values,-and-using-its-RMSE-to-compute-the-error-=-(aka-Maximum-Likelihood-Estimation)" data-toc-modified-id="Option-#1---Training-a-machine-learning-model-to-predict-values,-and-using-its-RMSE-to-compute-the-error-=-(aka-Maximum-Likelihood-Estimation)-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Option #1 - Training a machine learning model to predict values, and using its RMSE to compute the error = (aka Maximum Likelihood Estimation)</a></span></li><li><span><a href="#Option-#2---Training-a-machine-learning-model-to-predict-values-and-errors" data-toc-modified-id="Option-#2---Training-a-machine-learning-model-to-predict-values-and-errors-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Option #2 - Training a machine learning model to predict values and errors</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** An alternative to quantile regression useful for baseline model.
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
from sklearn.base import clone
rcParams['figure.figsize'] = 16, 8
rcParams['font.size'] = 20


# # Create a test function & dataset

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - To make the problem interesting, we generate observations of the target y as the sum of a deterministic term computed by the function f and a random noise term that follows a centered log-normal. 
# - To make this even more interesting we consider the case where the amplitude of the noise depends on the input variable x (**heteroscedastic** noise).
# - The lognormal distribution is non-symmetric and long tailed: observing large outliers is likely but it is impossible to observe small outliers.    
# 
# </font>
# </div>

# In[2]:


def f(x):
    """The function to predict."""
    return 2*x + x * np.sin(x)

rng = np.random.RandomState(42)
X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
expected_y = f(X).ravel()


# In[3]:


sigma = 0.5 + X.ravel() / 10
noise = rng.lognormal(sigma=sigma) - np.exp(sigma ** 2 / 2)
y = expected_y + noise


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Create an evenly spaced evaluation set of input values spanning the [0, 10] range.
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T


# In[14]:


fig = plt.figure()
plt.plot(xx, f(xx), 'g-', lw =3, label = r'$f_{custom}$')
plt.plot(X_test, y_test, 'b.', markersize = 10, label = 'Test set')
plt.plot(X_train, y_train, 'rx', markersize = 10, lw = 1, label = 'Train set')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')
plt.show()


# In[15]:


# split the data into a train set further for validation purpouses
# Yes, I know the dataset is small!
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, shuffle = True) 


# # An alternative (with assumptions)

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Let's assume the predictions follow a normal distribution.
# - This is a strong **assumption** but it seems to be good enough so it is still commonly used.
# - We'll have two models: one for the the prediction and one for its error.
# - So your prediction is the mean of the gaussian distribution, whereas the error estimate is its standard deviation.
# - This can be implemented in two ways here referred to as option #1 & #2.
# - For both options we use the table below to get the percentiles.
# - **Remember that**: standard deviation is the square root of the variance. 
# - **Remember that**: the variance is the mean absolute error.
#     
# </font>
# </div>

# ![image.png](attachment:image.png)

# ## Option #1 - Training a machine learning model to predict values, and using its RMSE to compute the error = (aka Maximum Likelihood Estimation)

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - This model assumes that the standard deviation of the normal distribution is constant. 
# - Essentially, the error is constant for all the predictions.
# - Keep in mind that we have split the dataset in 3 sets: training (training2 + test2=validation) and test. 
# - aka Maximum Likelihood Estimation according to this Ref: https://www.godaddy.com/engineering/2020/01/10/better-prediction-interval-with-neural-network/
# 
# </font>
# </div>

# In[5]:


# base model can be any regression model
modelbase_mode = GradientBoostingRegressor()
modelbase_mode.fit(X1, y1)
base_prediction = modelbase_mode.predict(X2)
# Compute the RMSE value
error = mean_squared_error(base_prediction, y2) ** 0.5
# compute the mean and standard deviation of the distribution
mean = modelbase_mode.predict(X_test)
st_dev = error


# In[6]:


# This is a single (symmetric) value applied to all of the predictions!
st_dev


# In[7]:


predictions = pd.DataFrame()
predictions["mean"] = mean 
predictions["X_test"] = X_test
predictions["up"] = mean + 1.96 * st_dev
predictions["low"] = mean - 1.96 * st_dev 
predictions.sort_values(by = ['X_test'], inplace = True)


# In[8]:


predictions.head(5)


# In[9]:


fig = plt.figure()

plt.plot(xx, f(xx), 'g-', lw = 3, label = r'$f_{custom}$')
plt.plot(X_test, mean, 'rx', lw =3, label = r'$predictions$')

plt.plot(predictions["X_test"], predictions["up"], 'k-')
plt.plot(predictions["X_test"], predictions["low"], 'k-')

plt.fill_between(predictions["X_test"], predictions["low"], predictions["up"], alpha = 0.5,
                 label = '90% Predicted Interval', color = "yellow")

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()


# ## Option #2 - Training a machine learning model to predict values and errors

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - In practice, the error is **not** always constant (it depends on the features).
# - Bare in mind that we are addressing the fact it may not be constant, but what we get is **still** a symmetric estimate of the variabilty.
# - Therefore, as an improvement, we can fit a model to learn the error itself.
# - As before, the base model is learnt from the training data.
# - Then, a second model (the error model) is trained on a validation set to predict the squared difference between the predictions and the real values.
# - The standard deviation of the distribution is computed by taking the root of the error predicted by the error model.
# 
# </font>
# </div>

# In[10]:


# base_model can be any regression model, a 
# sklearn.ensemble.GradientBoostingRegressor for instance 
base_model = GradientBoostingRegressor()
error_model = GradientBoostingRegressor()
base_model.fit(X1, y1) 
base_prediction = base_model.predict(X2) 
# compute the prediction error vector on the validation set 
validation_error = (base_prediction - y2) ** 2 
error_model.fit(X2, validation_error) 
# compute the mean and standard deviation of the distribution 
mean = base_model.predict(X_test) 
st_dev = error_model.predict(X_test)**0.5


# In[11]:


# So we get a value which is still symmetric but different for each prediction.
st_dev


# In[12]:


predictions = pd.DataFrame()
predictions["mean"] = mean 
predictions["X_test"] = X_test
predictions["up"] = mean + 1.96 * st_dev
predictions["low"] = mean - 1.96 * st_dev 
predictions.sort_values(by = ['X_test'], inplace = True)


# In[13]:


fig = plt.figure()

plt.plot(xx, f(xx), 'g-', lw = 3, label = r'$f_{custom}$')
plt.plot(X_test, mean, 'rx', lw =3, label = r'$predictions$')

plt.plot(predictions["X_test"], predictions["up"], 'k-')
plt.plot(predictions["X_test"], predictions["low"], 'k-')

plt.fill_between(predictions["X_test"], predictions["low"], predictions["up"], alpha = 0.5,
                 label = '90% Predicted Interval', color = "yellow")

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()


# # Conclusion
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-danger">
# <font color=black>
# 
# - The second method seemd to be **more premising** than first.
# - We are **not** address the asymmetric nature of the prediction error.
# - Compared to the **quantile** loss function we are fitting only two models (option #2): prediction and error estimators.
# - When you want a different quantile we just use the table reported above.                                                                             
# - Qunatile regressor or not, you still have to tune the models. 
# - At the of the day, you'll end up with the same amount of work to do.
#                                                                              
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://medium.com/@qucit/a-simple-technique-to-estimate-prediction-intervals-for-any-regression-model-2dd73f630bcb<br>
# 
# </font>
# </div>

# In[ ]:




