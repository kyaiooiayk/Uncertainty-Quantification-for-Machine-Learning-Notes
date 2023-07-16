#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** PIs via XGBoost Gradient Boosting with Quantile Loss Function on Boston Housing dataset
# 
# <br></font>
# </div>

# # Some theoretical recalls

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - A **CONFIDENCE** interval quantifies the uncertainty on an estimated population variable, such as the mean or 
# standard deviation. It can be used to quantify the uncertainty of the estimated skill of a model.
# - A **PREDICTION** interval quantifies the uncertainty on a single observation estimated from the population. It
# can be used to quantify the uncertainty of a single forecast. 
# - **Quantile:** These are points (essentially scalar values) in your data below which a certain proportion of your data fall. Consider a normal distribution with a mean of 0. The 0.5 quantile, or 50th percentile, is 0. Half the data lie below 0. That’s the peak of the hump in the curve. The 0.95 quantile, or 95th percentile, is about 1.64.  
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


import numpy as np
import pandas as pd
from math import e
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_pinball_loss
from statistics import stdev
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets, ensemble
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from pprint import pprint
rcParams['figure.figsize'] = 25, 8
rcParams['font.size'] = 25


# # Helper functions

# In[2]:


def report(results, n_top=1):
    """
    Report the first n_top CV results.
    """
    for i in range(1, n_top + 1):
        print(i)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(candidate)
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.9f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))

    return np.flatnonzero(results['rank_test_score'] == 1)[0]


# In[3]:


# hard or smooth
whichOne = "smooth"
# https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7


def log_cosh_quantile(alpha):
    """
    LogCosh quantile is nothing more than a smooth quantile loss function.
    This funciotion is C^oo so C1 and C2 which is all we need.
    """
    def _log_cosh_quantile(y_true, y_pred):
        err = (y_pred - y_true)
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)

        if whichOne == "smooth":
            grad = np.tanh(err)
            hess = 1 / np.cosh(err)**2
        if whichOne == "hard":
            grad = np.where(err < 0, alpha, (1 - alpha))
            hess = np.ones_like(err)

        return grad, hess

    return _log_cosh_quantile


# # Dataset

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Let’s look at the well-known **Boston housing** dataset and try to create prediction intervals
# - We’ll use 400 samples for training, leaving 106 samples for test. 
# 
# <br></font>
# </div>

# In[4]:


boston = load_boston()
x_df = pd.DataFrame(boston.data, columns=boston.feature_names)
x_df.describe()


# In[5]:


y_df = pd.DataFrame(boston.target, columns=['Target'])
y_df.describe()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(x_df, y_df,
                                                    test_size=.2,
                                                    shuffle=True,
                                                    random_state=7)


# In[7]:


# This is essentially the validation set
x_train2, x_test2, y_train2, y_test2 = \
    train_test_split(X_train, y_train, test_size=.2,
                     random_state=7, shuffle=True)
print(len(x_train2))
print(len(X_train))


# In[8]:


y_test.info()


# # Modelling + no tuning

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - PIs with the Gradient Boosting Regressor. GB builds an additive model in a forward stage-wise 
# fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree
# is fit on the negative gradient of the given loss function. We'll fit 3 models:
#     - One for the **lower prediction**
#     - One for the **upper prediction**
#     - One for the **mid prediction** which predicts the mean. This what is done all the time.
#       
# - **IMPORTANT**: when we change the loss to quantile and choose alpha (the quantile), we’re able to get predictions 
# corresponding to percentiles. If we use lower and upper quantiles, we can produce an estimated range which is exactly 
# what we want. 
# - **CL** stands for confidence level which is neither CI or PI.
# 
# <br></font>
# </div>

# In[9]:


# Confidence Level
CL = 0.90
alphaUpper = CL + (1 - CL)/2
alphaLower = (1 - CL)/2
print(alphaUpper*100, alphaLower*100)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Reference on how to create a custom-made **evaluation_metric** in XGBoost: https://coderzcolumn.com/tutorials/machine-learning/xgboost-an-in-depth-guide-python
# - Reference on how the **pin ball loss fnction** works in sklearn: https://scikit-learn.org/dev/modules/generated/sklearn.metrics.mean_pinball_loss.html        
# 
# <br></font>
# </div>

# In[10]:


def meanPinBallLossLower(preds, dmat):
    """
    Custom-made mean pin Ball loss
    y_true = [1, 2, 3]
    mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    which retunrs 0.033333333
    """

    targets = dmat.get_label()
    loss = mean_pinball_loss(targets, preds, alpha=alphaLower)

    return "PinBallLoss", loss


def meanPinBallLossUpper(preds, dmat):
    """
    Custom-made mean pin Ball loss
    y_true = [1, 2, 3]
    mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    which retunrs 0.033333333
    """

    targets = dmat.get_label()
    loss = mean_pinball_loss(targets, preds, alpha=alphaUpper)

    return "PinBallLoss", loss


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - There are 3 sets: training, test and validation.
# - Validation set is further out from the training set, in this way we leave the test set absolutely unseen.
# - Why is it necessary we use the validation test label with **_2** while training? Because during this process the model is assesed via some test set.
# - Where do we take this test set? We could use the original one, but in this way we'd be passing (even if implicitly) some info to the model.
# - So instead to further split the training set in another train and test set (tagged as _2) to avoid this implicit data leakage.
# - **IMPORTANT**: since we are evaluation on two dataset because we want to see if there is a kink in the validation test set **we are not using** this flag: early_stopping_rounds = 250
#     
# <br></font>
# </div>

# In[11]:


# upper percentile
modelUp = XGBRegressor(objective=log_cosh_quantile(alphaUpper))
modelUp.fit(x_train2, y_train2,
            eval_set=[(x_train2, y_train2), (x_test2, y_test2)],
            eval_metric=meanPinBallLossUpper,
            verbose=False)
resultsUp = modelUp.evals_result()

# lower percentile
modelLow = XGBRegressor(objective=log_cosh_quantile(alphaLower))
modelLow.fit(x_train2, y_train2,
             eval_set=[(x_train2, y_train2), (x_test2, y_test2)],
             eval_metric=meanPinBallLossLower,
             verbose=False)
resultsLow = modelLow.evals_result()

# prediction
modelMid = XGBRegressor()
modelMid.fit(x_train2, y_train2,
             eval_set=[(x_train2, y_train2), (x_test2, y_test2)],
             eval_metric="rmse",
             verbose=False)
resultsMid = modelMid.evals_result()


# In[12]:


modelLow


# In[13]:


preds = modelMid.predict(X_train)

r2_s = r2_score(y_train, preds)
mse = mean_squared_error(y_train, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, preds)

print("MidModel [MSE]_train80%: ", mse)
print("MidModel [RMSE]_train80%: ", rmse)
print("MidModel [MAE]_train80%: ", mae)
print("MidModel [R2]_train80%: ", r2_s)


# In[14]:


preds = modelMid.predict(X_test)

r2_s = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)

print("MidModel [MSE]_test20%: ", mse)
print("MidModel [RMSE]_test20%: ", rmse)
print("MidModel [MAE]_test20%: ", mae)
print("MidModel [R2]_test20%: ", r2_s)


# ## Plotting learning curve with cross validation

# <div class="alert alert-info">
# <font color=black>
# 
# - Mean computing MSE, RMSE, MAE and R2 you have to use the ls as a loss function. 
# - If you compute the median for the mid_model and then try to compute the MSE, this will force you to compare median with mean which are the same thing only if the distribution is normal.
# - From the results shown below we see the test error is a bit too high? Are we overfitting? Plotting the learnig curve can help us.
# 
# **AN IMPORTANT NOTE ON THE FOLLOWING PLOTS**
# - As reported in this [reference](https://scikit-learn.org/stable/modules/learning_curve.html) the `learning_curve`  is a tool  that shows the validation and training score of an estimator **for varying numbers of training samples**. It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error.
# - This has nothing to do with the learning curve plotted for XGBoost against its boosting round!
# 
# </font>
# </div>

# In[15]:


# List of models we'd like to evaluate
models = []
models.append(('mid', modelMid))
models.append(('lower', modelLow))
models.append(('upper', modelUp))


# In[16]:


# I've noticed that 10-fold split is a popular setting for this small dataset
kfold = KFold(n_splits=10, shuffle=True, random_state=7)


for name, model in models:

    score = "none"
    score_name = "pinball"
    if name == "mid":
        score = "neg_mean_squared_error"
        scoreName = "MSE"
    if name == "lower":
        score = make_scorer(mean_pinball_loss,
                            alpha=alphaLower, greater_is_better=False)
        scoreName = "pinBall"
    if name == "upper":
        score = make_scorer(mean_pinball_loss,
                            alpha=alphaUpper, greater_is_better=False)
        scoreName = "pinBall"
    print(score)
    # print(model)

    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X_train,
                                                            y_train,
                                                            cv=kfold,
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(
                                                                0.1, 1, 20),
                                                            scoring=score)

    train_scores_mean = -1*np.mean(train_scores, axis=1)
    train_scores_std = -1*np.std(train_scores, axis=1)
    test_scores_mean = -1*np.mean(test_scores, axis=1)
    test_scores_std = -1*np.std(test_scores, axis=1)

    # Plot learning curve
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score +/- std")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="CV test score +/- std")

    ax.grid(which="major", linestyle='-', linewidth='1.0', color='k')
    ax.grid(which="minor", linestyle='--', linewidth='0.25', color='k')
    ax.tick_params(which='major', direction='in', length=10, width=2)
    ax.tick_params(which='minor', direction='in', length=6, width=2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.minorticks_on()

    ax.set_xlabel("train size")
    ax.set_ylabel(scoreName)

    plt.legend(loc="best")
    plt.grid()


# ## Plotting learning with no-cross validation

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - **KEEP IN MIND THAT:**: since we are evaluation on two dataset because we want to see if there is a kink in the validation test set **we are not using** this flag: early_stopping_rounds = 250
# - From the the figure below we can see some sign of overfitting for the lower model at around epoch No: 90.
# 
# <br></font>
# </div>

# In[17]:


# plot classification error
fig, ax = plt.subplots()
epochs = len(resultsMid['validation_0']['rmse'])
x_axis = range(0, epochs)
ax.plot(x_axis, resultsMid['validation_0']['rmse'], "k-", lw=3, label='Train')
ax.plot(x_axis, resultsMid['validation_1']['rmse'], "r-", lw=3, label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('Learnig curve mean')

ax.grid(which="major", linestyle='-', linewidth='1.0', color='k')
ax.grid(which="minor", linestyle='--', linewidth='0.25', color='k')
ax.tick_params(which='major', direction='in', length=10, width=2)
ax.tick_params(which='minor', direction='in', length=6, width=2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.minorticks_on()

plt.show()


# In[18]:


# plot classification error
fig, ax = plt.subplots()
epochs = len(resultsLow['validation_0']['PinBallLoss'])
x_axis = range(0, epochs)
ax.plot(x_axis, resultsLow['validation_0']
        ['PinBallLoss'], "k-", lw=3, label='Train')
ax.plot(x_axis, resultsLow['validation_1']
        ['PinBallLoss'], "r-", lw=3, label='Test')
ax.legend()
plt.ylabel('Pin ball loss')
plt.title('Learnig curve lower')

ax.grid(which="major", linestyle='-', linewidth='1.0', color='k')
ax.grid(which="minor", linestyle='--', linewidth='0.25', color='k')
ax.tick_params(which='major', direction='in', length=10, width=2)
ax.tick_params(which='minor', direction='in', length=6, width=2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.minorticks_on()

plt.show()


# In[19]:


# plot classification error
fig, ax = plt.subplots()
epochs = len(resultsUp['validation_0']['PinBallLoss'])
x_axis = range(0, epochs)
ax.plot(x_axis, resultsUp['validation_0']
        ['PinBallLoss'], "k-", lw=3, label='Train')
ax.plot(x_axis, resultsUp['validation_1']
        ['PinBallLoss'], "r-", lw=3, label='Test')
ax.legend()
plt.ylabel('Pin ball loss')
plt.title('Learnig curve upper')

ax.grid(which="major", linestyle='-', linewidth='1.0', color='k')
ax.grid(which="minor", linestyle='--', linewidth='0.25', color='k')
ax.tick_params(which='major', direction='in', length=10, width=2)
ax.tick_params(which='minor', direction='in', length=6, width=2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.minorticks_on()

plt.show()


# # Tuning gradient boosting

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - For the mid model we'll use the **MSE** and the baseline MSE_test is ~ 12
# - But for the quantile we'll have to use a different one in order to pbtain a correct calibration.
# - The **pinball loss** function is a metric used to assess the accuracy of a quantile forecast. 
#     
# <br></font>
# </div>

# ## Tuning for the mean -> MSE

# In[20]:


param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.04, 0.05, 0.06],
    'n_estimators': [100, 150, 500],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.1, 0.5, 1.0],
    "colsample_bytree": [0.1, 0.5, 1.0]
}


kfold = KFold(n_splits=10, shuffle=True, random_state=7)
rsearchMid = RandomizedSearchCV(estimator=XGBRegressor(early_stopping_rounds=10),
                                param_distributions=param_grid,
                                n_iter=300,
                                random_state=7,
                                scoring='neg_mean_squared_error',
                                cv=kfold,
                                verbose=True,
                                n_jobs=-1)

rsearchMid.fit(X_train, y_train)
#print("Best hyperparamters: ", rsearch.best_estimator_)
#print("CV MSE [best hence not mean]: ", rsearch.best_score_)
# pprint(rsearch.best_params_)
indexBestModel = report(rsearchMid.cv_results_, n_top=2)


# In[21]:


indexBestModelMid = report(rsearchMid.cv_results_, n_top=1)
rsearchMid.cv_results_['params'][indexBestModelMid]


# ## Tuning for the lower and upper (quantile) -> pinball loss

# In[22]:


pinballLower = make_scorer(
    mean_pinball_loss,
    alpha=alphaLower,
    greater_is_better=False)


kfold = KFold(n_splits=10, shuffle=True, random_state=7)
rsearchLow = RandomizedSearchCV(estimator=XGBRegressor(
    objective=log_cosh_quantile(alphaLower),
    early_stopping_rounds=10),
    param_distributions=param_grid,
    n_iter=300,
    random_state=7,
    scoring=pinballLower,
    cv=kfold,
    verbose=True,
    n_jobs=-1)


rsearchLow.fit(X_train, y_train)
print("-------------------------")
print("Quantile=" + str(alphaLower))
#print("Best hyperparamters: ", rsearch.best_estimator_)
#print("CV pinball [best hence not mean]: ", rsearch.best_score_)
report(rsearchLow.cv_results_)
indexBestModelLow = report(rsearchLow.cv_results_, n_top=1)


# In[23]:


pinballUpper = make_scorer(
    mean_pinball_loss,
    alpha=alphaUpper,
    greater_is_better=False)


kfold = KFold(n_splits=10, shuffle=True, random_state=7)
rsearchUp = RandomizedSearchCV(estimator=XGBRegressor(
    objective=log_cosh_quantile(alphaUpper),
    early_stopping_rounds=10),
    param_distributions=param_grid,
    n_iter=300,
    random_state=7,
    scoring=pinballUpper,
    cv=kfold,
    verbose=True,
    n_jobs=-1)


rsearchUp.fit(X_train, y_train)
print("-------------------------")
print("Quantile=" + str(alphaLower))
#print("Best hyperparamters: ", rsearch.best_estimator_)
#print("CV pinball [best hence not mean]: ", rsearch.best_score_)
report(rsearchUp.cv_results_)
indexBestModelUp = report(rsearchUp.cv_results_, n_top=1)


# ## Plotting learning curves for tuned model via CV

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Let us see how the learning curves look
# - Each models will get its own hyperparameters.
# 
# <br></font>
# </div>

# In[24]:


kfold = KFold(n_splits=10, shuffle=True, random_state=7)

# As usual
mid_model_tuned = XGBRegressor(n_estimators=rsearchMid.cv_results_['params'][indexBestModelMid]["n_estimators"],
                               max_depth=rsearchMid.cv_results_[
                                   'params'][indexBestModelMid]["max_depth"],
                               learning_rate=rsearchMid.cv_results_[
                                   'params'][indexBestModelMid]["learning_rate"],
                               min_child_weight=rsearchMid.cv_results_[
                                   'params'][indexBestModelMid]["min_child_weight"],
                               subsample=rsearchMid.cv_results_[
                                   'params'][indexBestModelMid]["subsample"],
                               colsample_bytree=rsearchMid.cv_results_['params'][indexBestModelMid]["colsample_bytree"])
# Quantiles
lower_model_tuned = XGBRegressor(objective=log_cosh_quantile(alphaLower),
                                 n_estimators=rsearchLow.cv_results_[
                                     'params'][indexBestModelLow]["n_estimators"],
                                 max_depth=rsearchLow.cv_results_[
                                     'params'][indexBestModelLow]["max_depth"],
                                 learning_rate=rsearchLow.cv_results_[
                                     'params'][indexBestModelLow]["learning_rate"],
                                 min_child_weight=rsearchLow.cv_results_[
                                     'params'][indexBestModelMid]["min_child_weight"],
                                 subsample=rsearchLow.cv_results_[
                                     'params'][indexBestModelMid]["subsample"],
                                 colsample_bytree=rsearchLow.cv_results_['params'][indexBestModelMid]["colsample_bytree"])

upper_model_tuned = XGBRegressor(objective=log_cosh_quantile(alphaUpper),
                                 n_estimators=rsearchUp.cv_results_[
                                     'params'][indexBestModelUp]["n_estimators"],
                                 max_depth=rsearchUp.cv_results_[
                                     'params'][indexBestModelUp]["max_depth"],
                                 learning_rate=rsearchUp.cv_results_[
                                     'params'][indexBestModelUp]["learning_rate"],
                                 min_child_weight=rsearchUp.cv_results_[
                                     'params'][indexBestModelMid]["min_child_weight"],
                                 subsample=rsearchUp.cv_results_[
                                     'params'][indexBestModelMid]["subsample"],
                                 colsample_bytree=rsearchUp.cv_results_['params'][indexBestModelMid]["colsample_bytree"])

models = []
models.append(('mid_tuned', mid_model_tuned))
models.append(('lower_tuned', lower_model_tuned))
models.append(('upper_tuned', upper_model_tuned))


for name, model in models:
    score = ""
    score_name = ""
    if name == "mid_tuned":
        score = "neg_mean_squared_error"
        scoreName = "MSE"
    if name == "lower_tuned":
        score = make_scorer(mean_pinball_loss,
                            alpha=alphaLower, greater_is_better=False)
        scoreName = "pinBall"
    if name == "upper_tuned":
        score = make_scorer(mean_pinball_loss,
                            alpha=alphaUpper, greater_is_better=False)
        scoreName = "pinBall"
    print(score)
    print(scoreName)

    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X_train,
                                                            y_train,
                                                            cv=kfold,
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(
                                                                0.1, 1, 20),
                                                            scoring=score)

    train_scores_mean = -1*np.mean(train_scores, axis=1)
    train_scores_std = -1*np.std(train_scores, axis=1)
    test_scores_mean = -1*np.mean(test_scores, axis=1)
    test_scores_std = -1*np.std(test_scores, axis=1)

    # Plot learning curve
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score +/- std")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="CV test score +/- std")

    ax.grid(which="major", linestyle='-', linewidth='1.0', color='k')
    ax.grid(which="minor", linestyle='--', linewidth='0.25', color='k')
    ax.tick_params(which='major', direction='in', length=10, width=2)
    ax.tick_params(which='minor', direction='in', length=6, width=2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.minorticks_on()

    ax.set_xlabel("train size")
    ax.set_ylabel(scoreName)

    plt.legend(loc="best")
    plt.grid()
    plt.show()


# In[25]:


lower_model_tuned.fit(X_train, y_train)
mid_model_tuned.fit(X_train, y_train)
upper_model_tuned.fit(X_train, y_train)


# # Make predictions

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - With the models all trained, we now make predictions and record them with the true values.
# - We also adding the prediction obtained tuning the model, because we wanted to verify if the lack of tuning was causing some issue.
# 
# <br></font>
# </div>

# In[26]:


predictions = pd.DataFrame()
predictions["ID"] = np.arange(len(X_test))
predictions['truth'] = [i[0] for i in y_test.values.tolist()]
predictions['lower_tuned'] = lower_model_tuned.predict(X_test)
predictions['lower'] = modelLow.predict(X_test)
predictions['mid_tuned'] = mid_model_tuned.predict(X_test)
predictions['mid'] = modelMid.predict(X_test)
predictions['upper_tuned'] = upper_model_tuned.predict(X_test)
predictions['upper'] = modelUp.predict(X_test)
predictions["inOrOut"] = (predictions['upper'] > predictions['mid']) & (
    predictions['mid'] > predictions['lower'])
predictions["inOrOut_tuned"] = (predictions['upper_tuned'] > predictions['mid_tuned']) & (
    predictions['mid_tuned'] > predictions['lower_tuned'])
predictions["width_tuned"] = abs(
    predictions['upper_tuned'] - predictions['lower_tuned'])
predictions["width"] = abs(predictions['upper'] - predictions['lower'])


# In[27]:


predictions.head(5)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The **first** sanity check is passed with no problem
# - The **second** sanity check is not. There are some instances where the median > upper quantile.
# - In the section below we'll see how the optimised model does not solve this issue.
# - **However**, please look at the width of PI and the CWC metrics. These two show how the model has indeed improved.
# 
# <br></font>
# </div>

# In[28]:


predictions[predictions["inOrOut"] == False]


# In[29]:


predictions[predictions["inOrOut_tuned"] == False]


# In[30]:


fig, ax = plt.subplots()

plt.title("truth vs. prediction (_tuned and not)")
dummy = range(len(predictions["mid"].values))
ax.plot(dummy, predictions["truth"].values, "-r", lw=3, label="truth")
ax.plot(dummy, predictions["mid_tuned"].values, "--k", lw=3, label="mid_tuned")
ax.plot(dummy, predictions["mid"].values, "--k", lw=3, label="mid")

plt.legend()
plt.show()


# In[31]:


fig, ax = plt.subplots()

plt.title("truth vs. upper/lower prediction (_tuned and not)")
#dummy = range(len(predictions["mid_tuned"].values))
ax.plot(dummy, predictions["mid_tuned"].values, "-r", lw=3, label="mid_tuned")
ax.plot(dummy, predictions["upper_tuned"].values,
        "--k", lw=3, label="upper_tuned")
ax.plot(dummy, predictions["lower_tuned"].values,
        "--g", lw=3, label="lower_tuned")


plt.legend()
plt.show()


# In[32]:


fig, ax = plt.subplots()

plt.title("tune with prediction interval")
ax.plot(dummy, predictions["lower"].values, "--b", lw = 3, label = "lower")
ax.plot(dummy, predictions["lower_tuned"].values, "-b", lw = 3, label = "lower_tuned")
ax.plot(dummy, predictions["truth"].values, "-r", lw = 3, label = "truth")
ax.plot(dummy, predictions["upper"].values, "--g", lw = 3, label = "upper")
ax.plot(dummy, predictions["upper_tuned"].values, "-g", lw = 3, label = "upper_tuned")

ax.set_xlabel("x-test")
ax.set_ylabel("house price")
plt.legend()
plt.show()


# In[33]:


fig, ax = plt.subplots()

plt.title("Final result")
ax.plot(dummy, predictions["mid_tuned"].values, "-k", lw=3, label="mid_tuned")
ax.plot(dummy, predictions["lower_tuned"].values,
        "-g", lw=3, label="lower_tuned")
ax.plot(dummy, predictions["truth"].values, "--r", lw=2, label="truth")
ax.plot(dummy, predictions["upper_tuned"].values,
        "b-", lw=3, label="upper_tuned")

plt.fill_between(dummy, predictions["lower_tuned"].values, predictions["upper_tuned"].values, color="y", alpha=0.4,
                 label='CL=' + str(CL*100)+"%")

ax.set_xlabel("x-test")
ax.set_ylabel("house price")
plt.legend()
plt.show()


# In[34]:


fig, ax = plt.subplots()

for i in range(len(predictions["ID"])):
    ax.plot(predictions["ID"], predictions["truth"], "ro")
    mean = predictions["mid_tuned"][i]
    lower = mean - predictions["lower_tuned"][i]
    upper = predictions["upper_tuned"][i] - mean
    ax.errorbar(predictions["ID"][i], [mean], yerr=np.array([[lower, upper]]).T, fmt='bo-',
                solid_capstyle='projecting', capsize=5)

    # ax.legend()
plt.show()


# # PI coverage probability (PICP) & MPIW (Mean PI width)

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - [More important] **PI coverage probability (PICP)** is measured by counting the number of target values covered by the constructed PIs, where where ntest is the number of samples in the test set 
# - [Less important] PICP has a direct relationship with the width of PIs. A satisfactorily large PICP can be easily achieved by widening PIs from either side. However, such PIs are too conservative and less useful in practice, as they do not show the variation of the targets. Therefore, a measure is required to check how wide the PIs are. **Mean PI width (MPIW)** quantifies this aspect. 
# - The quality of PIs in this paper is assessed using the CWC.
# - As  CWC  covers  both  key  featuresof  PIs  (width  and  coverage  probability),  it  can  be  used  as the objective function to be minimised.
# - This means the **lower CWC is the better???**
# - Reference about CWC: Khosravi, Abbas, et al. "Comprehensive review of neural network-based prediction intervals and new advances." IEEE Transactions on neural networks 22.9 (2011): 1341-1356.
# - PIs  are  constructed  with  an  associated  90%  confidence level  (α equal  to  0.1). 
# - η and μ are  set  to  50  and  0.9.
# 
# <br></font>
# </div>

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[35]:


print("PICP      : ", len(
    predictions[predictions['inOrOut'] == True]) / len(X_test) * 100)
print("MPIW      : ", np.sum(predictions['width'].values) / len(X_test))

print("PICP_tuned: ", len(
    predictions[predictions['inOrOut_tuned'] == True]) / len(X_test) * 100)
print("MPIW_tuned: ", np.sum(predictions['width_tuned'].values) / len(X_test))


# In[36]:


nu = 0.9
eta = 50
gamma = 0
PICP = len(predictions[predictions['inOrOut'] == True]) / len(X_test)
MPIW = np.sum(predictions['width'].values) / len(X_test)
delta = predictions['width'].max() - predictions['width'].min()
NMPIW = np.sum(predictions['width'].values) / len(X_test) / delta

if PICP < nu:
    gamma = 1

CWC = NMPIW * (1 + gamma * PICP * e**(- eta * (PICP - nu)))
print("CWC_notTuned=", CWC)


# In[37]:


nu = CL
gamma = 0
PICP = len(predictions[predictions['inOrOut_tuned'] == True]) / len(X_test)
MPIW = np.sum(predictions['width_tuned'].values) / len(X_test)
delta = predictions['width_tuned'].max() - predictions['width'].min()
NMPIW = np.sum(predictions['width_tuned'].values) / len(X_test) / delta

if PICP < nu:
    gamma = 1

CWC = NMPIW * (1 + gamma * e**(- eta * (PICP - nu)))
print("CWC_tuned=", CWC)


# # Conclusions
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-danger">
# <font color=black><br>
# 
# - The Boston dataset is challenging because it has very little data ~500 pts.
# - There are 3 models to tune: mean, upper and lowe qunatiles.
# - The mean model is nothing more than the one you obtain normally.
# - Each of the 3 model needs to be tuned separately and this is almost certainly going to kill oof any application for very big DL models.
# - PICP & MIPW are two PI metrics for which you have to find a tradeoff.
# - **CWC** and the way it was constructed does not convince me at all. Are there better formulation?
# 
# <br></font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# - https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed<br>
# - https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb<br>
# - https://scikit-learn.org/dev/auto_examples/ensemble/plot_gradient_boosting_quantile.html<br>
# 
# <br></font>
# </div>

# In[ ]:




