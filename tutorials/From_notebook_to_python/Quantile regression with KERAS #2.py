#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Quantile regression with KERAS
# 
# **Reference [article]:** https://towardsdatascience.com/deep-quantile-regression-c85481548b5a<br>
# **Reference [code]:** https://github.com/sachinruk/KerasQuantileModel/blob/master/Keras%20Quantile%20Model.ipynb<br>
# 
# <br></font>
# </div>

# # Theoretical recall

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The quantile regression loss function is applied to predict quantiles. 
# - Our goal would be to minimise this function.
# - A quantile is the value below which a fraction of observations in a group falls.
# - Good to keep in mind: MAE = Mean Absolute Error is the same as q = 0.5.
# - Uncertainty and quantiles are **not** the same thing. 
# - But most of the time you **care about** quantiles and not uncertainty.
# - 0f you really do want **uncertainty** with deep nets checkout http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
#     
# <br></font>
# </div>

# # Imort modules

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib import rcParams
import keras

rcParams['figure.figsize'] = 14, 6
rcParams['font.size'] = 20


# # Import dataset

# In[4]:


mcycle = pd.read_csv('../DATASETS/motorcycleAccident.txt', delimiter = '\t')


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Standardise the inputs and outputs so that it is easier to train.
# - So apparently the guy is standardisining the input for regression, something that I was wondering about but I have not found something to back it up.
# 
# <br></font>
# </div>

# In[ ]:


mcycle.times = (mcycle.times - mcycle.times.mean())/mcycle.times.std()
mcycle.accel = (mcycle.accel - mcycle.accel.mean())/mcycle.accel.std()


# # Building of the model for the MEAN

# In[51]:


model = Sequential()
model.add(Dense(units = 10, input_dim=1, activation = 'relu'))
model.add(Dense(units = 10, input_dim=1, activation = 'relu'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(loss = 'mae', optimizer = opt)


# In[52]:


model.fit(mcycle.times.values, mcycle.accel.values, epochs = 2000, batch_size = 32, verbose=1)
model.evaluate(mcycle.times.values, mcycle.accel.values)


# ## Post-processing - mean results

# In[54]:


t_test = np.linspace(mcycle.times.min(),mcycle.times.max(),200)
y_test_mean = model.predict(t_test)

plt.scatter(mcycle.times,mcycle.accel, label = "truth")
plt.plot(t_test, y_test_mean, 'r-', label = "predictions [mean]")

plt.legend()
plt.show()


# # Building the model for the quantile

# In[26]:


def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


# In[56]:


def mcycleModel():
    model = Sequential()
    model.add(Dense(units=10, input_dim=1,activation='relu'))
    model.add(Dense(units=10, input_dim=1,activation='relu'))
    model.add(Dense(units=10, input_dim=1,activation='relu'))
    model.add(Dense(1))
    
    return model


# ## Lower quantile

# In[57]:


modelLow = mcycleModel()
optQuantile = keras.optimizers.Adam(learning_rate = 0.005)
modelLow.compile(loss=lambda y,f: tilted_loss(0.05,y,f), optimizer = optQuantile)
modelLow.fit(mcycle.times.values, mcycle.accel.values, epochs = 2000, batch_size=32, verbose=0)

# Predict the quantile
y_test_low = modelLow.predict(t_test)


# ## Upper quantile

# In[58]:


modelUp = mcycleModel()
optQuantile = keras.optimizers.Adam(learning_rate = 0.005)
modelUp.compile(loss=lambda y,f: tilted_loss(0.9,y,f), optimizer = optQuantile)
modelUp.fit(mcycle.times.values, mcycle.accel.values, epochs = 2000, batch_size=32, verbose=0)

# Predict the quantile
y_test_up = modelUp.predict(t_test)


# In[59]:


t_test = np.linspace(mcycle.times.min(),mcycle.times.max(),200)
plt.scatter(mcycle.times,mcycle.accel)

plt.plot(t_test, y_test_low, "g-", label = "lower") 
plt.plot(t_test, y_test_mean, 'r-' , label = "mean")
plt.plot(t_test, y_test_up, "b-", label = "upper") 

plt.legend()    
plt.show()



# # Conclusion

# <div class="alert alert-block alert-danger">
# <font color=black><br>
# 
# - There should be a way to compute the quantile in one go withouth having to re-run the model twice (lower & upper models).
# - This does seems to be an easy thing to do in KERAS as exaplined in the article.
# - However you can do it in tensor flow as shown in here: https://github.com/strongio/quantile-regression-tensorflow/blob/master/Quantile%20Loss.ipynb
# 
# <br></font>
# </div>

# In[ ]:




