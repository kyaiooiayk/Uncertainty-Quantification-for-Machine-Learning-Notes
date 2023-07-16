#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Classification-vs.-regression" data-toc-modified-id="Classification-vs.-regression-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Classification vs. regression</a></span></li><li><span><a href="#Multi-quantile-loss-function" data-toc-modified-id="Multi-quantile-loss-function-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Multi-quantile loss function</a></span></li><li><span><a href="#Simple-Linear-Regression" data-toc-modified-id="Simple-Linear-Regression-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Simple Linear Regression</a></span></li><li><span><a href="#Coverage" data-toc-modified-id="Coverage-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Coverage</a></span></li><li><span><a href="#Non-Linear-Regression-with-Variable-Noise" data-toc-modified-id="Non-Linear-Regression-with-Variable-Noise-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Non-Linear Regression with Variable Noise</a></span></li><li><span><a href="#Folder-clean-up" data-toc-modified-id="Folder-clean-up-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Folder clean-up</a></span></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Conclusions</a></span></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Multi-quantile loss with CatBoost
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
sns.set()


# In[2]:


import catboost
catboost.__version__


# # Classification vs. regression
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - While many classification models, particularly calibrated models, come with uncertainty quantification by predicting a probability distribution over target classes, quantifying uncertainty in regression tasks is much more nuanced.
# 
# </font>
# </div>

# # Multi-quantile loss function 
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - **Classical quantile regression**: the main disadvantage of quantile regression was that one model had to be trained per predicted quantile. For instance, in order to predict the 10th, 50th, and 90th quantiles of a target distribution, three independent models would need to be trained. 
# - Catboost has since addressed this issue with the **multi-quantile loss function** — a loss function that enables a single model to predict an arbitrary number of quantiles. Catboost now extends this idea by allowing the base decision trees to output multiple quantiles per node. This allows a single model to predict multiple quantiles by minimizing a new single loss function.
#     
# </font>
# </div>

# # Simple Linear Regression
# <hr style="border:2px solid black"> </hr>

# In[3]:


# Number of training and testing examples
n = 1000

# Generate random x values between 0 and 1
x_train = np.random.rand(n)
x_test = np.random.rand(n)

# Generate random noise for the target
noise_train = np.random.normal(0, 0.3, n)
noise_test = np.random.normal(0, 0.3, n)

# Set the slope and y-intercept of the line
a, b = 2, 3

# Generate y values according to the equation y = ax + b + noise
y_train = a * x_train + b + noise_train
y_test = a * x_test + b + noise_test


# In[4]:


# Store quantiles 0.01 through 0.99 in a list
quantiles = [q/100 for q in range(1, 100)]

# Format the quantiles as a string for Catboost
quantile_str = str(quantiles).replace('[','').replace(']','')

# Fit the multi quantile model
model = CatBoostRegressor(iterations=100,
                          loss_function=f'MultiQuantile:alpha={quantile_str}')

model.fit(x_train.reshape(-1,1), y_train)


# In[5]:


# Make predictions on the test set
preds = model.predict(x_test.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=[f'pred_{q}' for q in quantiles])


# In[6]:


preds


# In[7]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_test, y_test)

for col in ['pred_0.05', 'pred_0.5', 'pred_0.95']:
    ax.scatter(x_test.reshape(-1,1), preds[col], alpha=0.50, label=col)

ax.legend()


# # Coverage
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - When working with predicted quantiles, one **metric** we’re often interested in analyzing is coverage.
# - **Coverage** is the percentage of targets that fall between two desired quantiles. As an example, coverage can be computed using the 5th and 95th quantiles as follows:
# - Using the 5th and 95th quantiles, **assuming perfect calibration**, we would expect to have a coverage of 95–5 = 90%. In this example, the predicted quantiles were slightly off but still close, giving a coverage value of 91.4%.
#     
# </font>
# </div>

# In[8]:


coverage_90 = np.mean((y_test <= preds['pred_0.95']) & (
    y_test >= preds['pred_0.05']))*100
print(coverage_90)


# # Non-Linear Regression with Variable Noise
# <hr style="border:2px solid black"> </hr>

# In[9]:


# Create regions of the domain that have variable noise
bounds = [(-10, -8), (-5, -4), (-4, -3), (-3, -1),
          (-1, 1), (1, 3), (3, 4), (4, 5), (8, 10)]
scales = [18, 15, 8, 11, 1, 2, 9, 16, 19]

x_train = np.array([])
x_test = np.array([])
y_train = np.array([])
y_test = np.array([])

for b, scale in zip(bounds, scales):

    # Randomly select the number of samples in each region
    n = np.random.randint(low=100, high=200)

    # Generate values of the domain between b[0] and b[1]
    x_curr = np.linspace(b[0], b[1], n)

    # For even scales, noise comes from an exponential distribution
    if scale % 2 == 0:

        noise_train = np.random.exponential(scale=scale, size=n)
        noise_test = np.random.exponential(scale=scale, size=n)

    # For odd scales, noise comes from a normal distribution
    else:

        noise_train = np.random.normal(scale=scale, size=n)
        noise_test = np.random.normal(scale=scale, size=n)

    # Create training and testing sets
    y_curr_train = x_curr**2 + noise_train
    y_curr_test = x_curr**2 + noise_test

    x_train = np.concatenate([x_train, x_curr])
    x_test = np.concatenate([x_test, x_curr])
    y_train = np.concatenate([y_train, y_curr_train])
    y_test = np.concatenate([y_test, y_curr_test])


# In[10]:


model = CatBoostRegressor(iterations=300,
                          loss_function=f'MultiQuantile:alpha={quantile_str}')

model.fit(x_train.reshape(-1, 1), y_train)

preds = model.predict(x_test.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=[f'pred_{q}' for q in quantiles])


# In[11]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_test, y_test)

for col in ['pred_0.05', 'pred_0.5', 'pred_0.95']:

    quantile = int(float(col.split('_')[-1])*100)
    label_name = f'Predicted Quantile {quantile}'
    ax.scatter(x_test.reshape(-1, 1), preds[col], alpha=0.50, label=label_name)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Testing Data for Example 2 with Predicted Quantiles')
ax.legend()


# <div class="alert alert-info">
# <font color=black>
#     
# - Upon visual inspection, the model has characterized this non-linear, heteroscedastic relationship well. Notice how, near x = 0, the three predicted quantiles converge towards a single value.
# - This is because the region near x = 0 has almost no noise — any model that correctly predicts the conditional probability distribution in this region should predict a small variance. Conversely, when x is between 7.5 and 10.0, the predicted quantiles are much further apart because of the inherent noise in the region. 90% coverage can be computed as before:
#     
# </font>
# </div>

# In[12]:


coverage_90 = np.mean((y_test <= preds['pred_0.95']) & (
    y_test >= preds['pred_0.05']))*100
print(coverage_90)


# # Folder clean-up
# <hr style="border:2px solid black"> </hr>

# In[13]:


get_ipython().system('rm -rf catboost_info')


# # Conclusions
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-danger">
# <font color=black>
#     
# - It’s important to note that quantile regression makes no statistical or algorithmic guarantees of convergence, and the performance of these models will vary depending on the nature of the learning problem. 
# 
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://towardsdatascience.com/a-new-way-to-predict-probability-distributions-e7258349f464
# - https://catboost.ai/en/docs/concepts/loss-functions-regression#MultiQuantile
# - https://brendanhasz.github.io/2018/12/15/quantile-regression.html
#     
# </font>
# </div>

# # Requirements
# <hr style="border:2px solid black"> </hr>

# In[14]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')


# In[ ]:




