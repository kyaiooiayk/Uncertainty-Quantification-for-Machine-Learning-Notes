#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Confidence interval via bootstrap
# 
# </font>
# </div>

# # Import modules
# <hr style="border:2px solid black"> </hr>

# In[1]:


import numpy
from pandas import read_csv
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


# # Theory recalls

# ## Why bootstrap?

# <div class="alert alert-info">
# <font color=black>
# 
# - A robust way to calculate confidence intervals for machine learning algorithms is to use the bootstrap. 
# - This is a general technique for estimating statistics that can be used to calculate empirical confidence intervals, regardless of the distribution (e.g. also non-Gaussian)
# 
# </font>
# </div>

# ## Calculate a population statistics

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The first step is to use the bootstrap procedure to resample the original data a number of times and calculate the statistic of interest. 
# - The dataset is sampled with replacement. This means that each time an item is selected from the original dataset, it is not removed, allowing that item to possibly be selected again for the sample. 
# - The number of bootstrap repeats defines the variance of the estimate, and more is better, often hundreds or thousands.
# - The pseudo code would look like this:
# 
# <br></font>
# </div>

# In[2]:


"""
statistics = []
for i in bootstraps:
    sample = select_sample_with_replacement(data)
    stat = calculate_statistic(sample)
    statistics.append(stat)
"""


# ## Calculate confidence interval

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Now that we have a population of the statistics of interest, we can calculate the confidence intervals.
# - This is done by first ordering the statistics, then selecting values at the chosen percentile for the confidence interval. The chosen percentile in this case is called alpha.
# - For example, if we were interested in a confidence interval of 95%, then alpha would be 0.95 and we would select  the value at the 2.5% percentile as the lower bound and the 97.5% percentile as the upper bound on the statistic of  interest. 2.5% comes from 1-0.95/2=5/2=2.5%; the division by two comes from the need to require both upper and  lower interval.
# - For example, if we calculated 1,000 statistics from 1,000 bootstrap samples, then the lower bound would be the 25th value and the upper bound would be the 97.5th value, assuming the list of statistics was ORDERED.
# - In this, we are calculating a non-parametric confidence interval that does not make any assumption about the  functional form of the distribution of the statistic. 
# - This confidence interval is often called the empirical confidence interval.
# 
# <br></font>
# </div>

# In[3]:


"""
ordered = sort(statistics)
lower = percentile(ordered, (1-alpha)/2)
upper = percentile(ordered, alpha+((1-alpha)/2))
"""


# # Calculate Classification Accuracy Confidence Interval

# In[4]:


# load dataset
data = read_csv('../DATASETS/pima-indians-diabetes.csv', header=None)
values = data.values


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Next, we will configure the bootstrap. 
# - We will use 1,000 bootstrap iterations and select a sample that is 50% the size of the dataset.
# 
# <br></font>
# </div>

# In[5]:


# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The sample will be selected with replacement using the resample() function from sklearn. 
# - Any rows that were not included in the sample are retrieved and used as the test dataset. 
# - Next, a decision tree classifier is fit on the sample and evaluated on the test set, a classification score calculated, and added to a list of scores collected  across all the bootstraps.
# 
# <br></font>
# </div>

# In[6]:


# run bootstrap
stats = list()
for i in range(n_iterations):
    # Prepare train and test sets
    train = resample(values, n_samples=n_size)
    test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
    # Fit model on the TRAINING data
    model = DecisionTreeClassifier()
    model.fit(train[:,:-1], train[:,-1])
    # Evaluate model on the TEST data
    predictions = model.predict(test[:,:-1])
    score = accuracy_score(test[:,-1], predictions)
    print(score)
    stats.append(score)    


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Once the scores are collected, a histogram is created to give an idea of the distribution of scores. 
# - We would generally expect this distribution to be Gaussian, perhaps with a skew with a symmetrical variance around the mean.
# - Please note that the bootstrapping procedure works also for non-Gaussian method.
# - Finally, we can calculate the empirical confidence intervals using the percentile() NumPy function. 
# - A 95% confidence interval is used, so the values at the 2.5 and 97.5 percentiles are selected.
# 
# <br></font>
# </div>

# In[7]:


# plot scores
pyplot.hist(stats)
pyplot.show()


# In[16]:


# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(stats, p))
print('%.1f confidence interval with lower and upper percentile at %.3f and %.3f respectively' % (alpha*100, lower, upper))


# <div class="alert alert-info">
# <font color=black>
# 
# - Finally, the confidence intervals are reported, showing that there is a 95% likelihood that the confidence interval 64.4% and 73.0% covers the true skill of the model. 
# - This same method can be used to calculate confidence intervals of any other errors scores, such as root mean squared error for regression algorithms.
# 
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python
# - https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
#     
# </font>
# </div>

# In[ ]:




