#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Plotting confidence interval
# 
# </font>
# </div>

# # Import modules

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# Getting rid of the warning messages
import warnings
warnings.filterwarnings("ignore")


# # Method #1 - lineplot

# In[2]:


"""
Variable x will store 100 random integers from 0 (inclusive) to 30 (exclusive) and variable y will store 100 
samples from the Gaussian (Normal) distribution which is centred at 0 with spread/standard deviation 1. 
Finally, a lineplot is created with the help of seaborn library with 95% confidence interval by default.
The light blue shade indicates the confidence level around that point if it has higher confidence the shaded 
line will be thicker.
"""


# In[3]:


# generate random data
np.random.seed(0)
x = np.random.randint(0, 30, 100)
y = x+np.random.normal(0, 1, 100)

# create lineplot
ax = sns.lineplot(x, y, ci=95)


# # Method #2 - regplot

# In[4]:


"""
Basically, it includes a regression line in the scatterplot and helps in seeing any linear relationship between 
two variables. Below example will show how it can be used to plot confidence interval as well.
"""


# In[5]:


# create random data
np.random.seed(0)
x = np.random.randint(0, 10, 10)
y = x+np.random.normal(0, 1, 10)

# create regression plot
ax = sns.regplot(x, y, ci=80)


# # Method #3 - Boostrapping

# In[6]:


"""
Bootstrapping is a test/metric that uses random sampling with replacement. It gives the measure of accuracy 
(bias, variance, confidence intervals, prediction error, etc.) to sample estimates. It allows the estimation 
of the sampling distribution for most of the statistics using random sampling methods. It may also be used for 
constructing hypothesis tests. 
"""


# In[7]:


# load dataset
x = np.array([180, 162, 158, 172, 168, 150, 171, 183, 165, 176])

# configure bootstrap
n_iterations = 1000  # here k=no. of bootstrapped samples
n_size = int(len(x))

# run bootstrap
medians = list()
for i in range(n_iterations):
    s = resample(x, n_samples=n_size)
    m = np.median(s)
    medians.append(m)

# plot scores
plt.hist(medians)
plt.show()

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = np.percentile(medians, p)
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = np.percentile(medians, p)

print(f"\n{alpha*100} confidence interval {lower} and {upper}")


# In[8]:


"""
After importing all the necessary libraries create a sample S with size n=10 and store it in a variable x. 
Using a simple loop generate 1000 artificial samples (=k) with each sample size m=10 (since m<=n). These samples
are called the bootstrapped sample. Their medians are computed and stored in a list ‘medians’. Histogram of Medians 
from 1000 bootstrapped samples is plotted with the help of matplotlib library and using the formula confidence 
interval of a sample statistic calculates an upper and lower bound for the population value of the statistic at a 
specified level of confidence based on sample data is calculated
"""


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.geeksforgeeks.org/how-to-plot-a-confidence-interval-in-python/
# 
# </font>
# </div>

# In[ ]:




