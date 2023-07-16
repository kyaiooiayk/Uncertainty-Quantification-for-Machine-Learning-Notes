#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Prediction intervals with gradient boosting
# 
# <br></font>
# </div>

# # Theortical recalls

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - **A CONFIDENCE interval** quantifies the uncertainty on an estimated population variable, such as the mean or 
# standard deviation. It can be used to quantify the uncertainty of the estimated skill of a model.
# - **A PREDICTION interval** quantifies the uncertainty on a single observation estimated from the population. It
# can be used to quantify the uncertainty of a single forecast. 
# 
# <br></font>
# </div>

# # Project's goal

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Prediction building energy consumption and the associated prediction interval for each prediction. 
# - There are undoubtedly hidden features (latent variables) not captured in our data that affect energy consumption.
# - **THEREFORE**, we want to show the uncertainty in our estimates by predicting both an upper and lower bound for energy use.
# 
# <br></font>
# </div>

# # Import modules

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
import glob
from ipywidgets import interact, widgets
import plotly.graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly_express as px
import cufflinks as cf


# # Check packages version

# In[ ]:


import numpy,pandas,plotly
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-p numpy,pandas,plotly')


# # Dataset

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The energy data is measured every 15 minutes and includes 3 weather variables related to energy consumption: temperature, irradiance, and relative humidity. 
# - This is the data from the DrivenData Energy Forecasting competition.
# - I've cleaned up the datasets and extracted 8 features that allow us to predict the energy consumption fairly accurately.
# - Dataset refrence: https://www.drivendata.org/competitions/51/electricity-prediction-machine-learning/
# 
# <br></font>
# </div>

# In[ ]:


files = glob.glob('../DATASETS/*_energy_data.csv')
files


# In[ ]:


data = pd.read_csv(files[2], parse_dates=['timestamp'],
                   index_col='timestamp').sort_index()
data.head()
data = data.rename(columns={"energy": "actual"})


# In[ ]:


data.head(5)


# In[ ]:


# Create a subset of data for plotting
data_to_plot = data.loc["2015"].copy()


def plot_timescale(timescale, selection, theme):
    """
    Plot the energy consumption on different timescales (day, week, month).
    
    :param timescale: the timescale to use
    :param selection: the numeric value of the timescale selection (for example the 15th day
    of the year or the 1st week of the year)
    :param theme: aesthetics of plot
    """
    # Subset based on timescale and selection
    subset = data_to_plot.loc[
        getattr(data_to_plot.index, timescale) == selection, "actual"
    ].copy()

    if subset.empty:
        print("Choose another selection")
        return
    
    # Make an interactive plot
    fig = subset.iplot(
            title=f"Energy for {selection} {timescale.title()}", theme=theme, asFigure=True
    )
    fig['layout']['height'] = 500
    fig['layout']['width'] = 1400
    iplot(fig)
    


_ = interact(
    plot_timescale,
    timescale=widgets.RadioButtons(
        options=["dayofyear", "week", "month"], value="dayofyear"
    ),
    # Selection 
    selection=widgets.IntSlider(value=16, min=0, max=365),
    theme=widgets.Select(options=cf.themes.THEMES.keys(), value='ggplot')
)


# In[ ]:


data.loc['2015-01-01':'2015-07-01', "actual"].iplot(layout=dict(title='2015 Energy Consumption', height=500))


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Clearly, there are different patterns in energy usage over the course of a day, week, and month. 
# - We can also look at longer timescales. 
# - Plotting this much data can make the notebook slow. Instead, we can resample the data and plot to see any long term trends.
# 
# <br></font>
# </div>

# In[ ]:


data.resample('12 H')["actual"].mean().iplot(layout=dict(title='Energy Data Resampled at 12 Hours', height=500,
                                                        yaxis=dict(title='kWh')))


# # Modelling

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Predicting Intervals with the Gradient Boosting Regressor. GB builds an additive model in a forward stage-wise 
# fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree
# is fit on the negative gradient of the given loss function. From sklearn userguide we know these are the options:
# loss{‘ls’, ‘lad’, ‘huber’, ‘quantile’}.
# 
#     - For the lower prediction, use the GradientBoostingRegressor with loss='quantile' and alpha=lower_quantile 
#       (for example, 0.1 for the 10th percentile)
#     - For the upper prediction, use the GradientBoostingRegressor with loss='quantile' and alpha=upper_quantile 
#       (for example, 0.9 for the 90th percentile)
#     - For the mid prediction, use GradientBoostingRegressor(loss="quantile", alpha=0.5) which predicts the median,
#       or the default loss="ls" (for least squares) which predicts the mean which was we are going to use 
#       
# - **IMPORTANT POINT:** when we change the loss to quantile and choose alpha (the quantile), we’re able to get predictions 
# corresponding to percentiles. If we use lower and upper quantiles, we can produce an estimated range which is exactly 
# what we want.   
# 
# <br></font>
# </div>

# In[ ]:


# Train and test sets
X_train = data.loc["2015":"2016"].copy()
X_test = data.loc["2017":].copy()
y_train = X_train.pop("actual")
y_test = X_test.pop("actual")

assert X_train.index.max() < X_test.index.min()


# In[ ]:


X_train.tail()


# In[ ]:


X_test.head()


# In[ ]:


# Set lower and upper quantile
LOWER_ALPHA = 0.15
UPPER_ALPHA = 0.85

N_ESTIMATORS = 100
MAX_DEPTH = 5


lower_model = GradientBoostingRegressor(loss = "quantile", alpha=LOWER_ALPHA, n_estimators=N_ESTIMATORS, 
                                        max_depth=MAX_DEPTH)

# The mid model will use the default ls which predict the mean
mid_model = GradientBoostingRegressor(loss = "ls", n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)

upper_model = GradientBoostingRegressor(loss = "quantile", alpha=UPPER_ALPHA, n_estimators=N_ESTIMATORS, 
                                        max_depth=MAX_DEPTH)


# # Training

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Build Models for Lower, Upper Quantile and Mean. Remember our goal is to get the interval, at the moment we are 
# not concentrated on the getting the best, hence there is no hyperparameters tuning here.
# - The models are trained based on optimizing for the specific loss function. 
# - This means we have to build 3 separate models to predict the different objectives. 
# - A downside of this method is that it's a little slow, particularly because we can't parallelize training on the Scikit-Learn Gradient Boosting Regresssor. 
# - If you wanted, you could re-write this code to train each model on a separate processor (using multiprocessing.)
# 
# <br></font>
# </div>

# In[ ]:


_ = lower_model.fit(X_train, y_train)
_ = mid_model.fit(X_train, y_train)
_ = upper_model.fit(X_train, y_train)


# # Make predictions

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# -  With the models all trained, we now make predictions and record them with the true values.
# 
# <br></font>
# </div>
# 

# In[ ]:


predictions = pd.DataFrame(y_test)
predictions['lower'] = lower_model.predict(X_test)
predictions['mid'] = mid_model.predict(X_test)
predictions['upper'] = upper_model.predict(X_test)

assert (predictions['upper'] > predictions['lower']).all()


# In[ ]:


predictions.tail()


# # Prediction Intervals Plot

# In[ ]:


def plot_intervals(predictions, mid=False, start=None, stop=None, title=None):
    """
    Function for plotting prediction intervals as filled area chart.
    
    :param predictions: dataframe of predictions with lower, upper, and actual columns (named for the target)
    :param whether to show the mid prediction
    :param start: optional parameter for subsetting start of predictions
    :param stop: optional parameter for subsetting end of predictions
    :param title: optional string title
    
    :return fig: plotly figure
    """
    # Subset if required
    predictions = (
        predictions.loc[start:stop].copy()
        if start is not None or stop is not None
        else predictions.copy()
    )
    data = []

    # Lower trace will fill to the upper trace
    trace_low = go.Scatter(
        x=predictions.index,
        y=predictions["lower"],
        fill="tonexty",
        line=dict(color="darkblue"),
        fillcolor="rgba(173, 216, 230, 0.4)",
        showlegend=True,
        name="lower",
    )
    # Upper trace has no fill
    trace_high = go.Scatter(
        x=predictions.index,
        y=predictions["upper"],
        fill=None,
        line=dict(color="orange"),
        showlegend=True,
        name="upper",
    )

    # Must append high trace first so low trace fills to the high trace
    data.append(trace_high)
    data.append(trace_low)
    
    if mid:
        trace_mid = go.Scatter(
        x=predictions.index,
        y=predictions["mid"],
        fill=None,
        line=dict(color="green"),
        showlegend=True,
        name="mid",
    )
        data.append(trace_mid)

    # Trace of actual values
    trace_actual = go.Scatter(
        x=predictions.index,
        y=predictions["actual"],
        fill=None,
        line=dict(color="black"),
        showlegend=True,
        name="actual",
    )
    data.append(trace_actual)

    # Layout with some customization
    layout = go.Layout(
        height=500,
        width=1400,
        title=dict(text="Prediction Intervals" if title is None else title),
        yaxis=dict(title=dict(text="kWh")),
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    # Make sure font is readable
    fig["layout"]["font"] = dict(size=20)
    fig.layout.template = "plotly_white"
    return fig


# Example plot subsetted to one week
fig = plot_intervals(predictions, start="2017-03-01", stop="2017-03-08")


# In[ ]:


# Interactive plotting
iplot(fig)


# # Calculate the errors

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Quantifying the error of a prediction range can be tricky. We'll start off with the percentage of the time that the 
# actual value falls in the range. 
# - However, one way to maximize this metric would be to just use extremely wide prediction intervals. 
# - Therefore, we want to penalize the model for making too wide prediction intervals. 
# - As a simple example we can calculate the absolute error of the bottom and top lines, and divide by two to get an absolute error. We then take the average for the mean absolute error. 
# - We can also calculate the absolute error of the mid predictions.
# - These are likely not the best metrics for all cases.
# 
# <br></font>
# </div>

# In[ ]:


def calculate_error(predictions):
    """
    Calculate the absolute error associated with prediction intervals
    
    :param predictions: dataframe of predictions
    :return: None, modifies the prediction dataframe    
    """
    
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["actual"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["actual"]).abs()
    
    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["actual"]).abs()
    
    predictions['in_bounds'] = predictions["actual"].between(left=predictions['lower'], right=predictions['upper'])


# In[ ]:


calculate_error(predictions)
metrics = predictions[['absolute_error_lower', 'absolute_error_upper', 'absolute_error_interval', 'absolute_error_mid', 'in_bounds']].copy()
metrics.describe()


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We see the lower prediction has a smaller absolute error (in terms of the median). 
# - It's interesting the absolute error for the lower bound is actually less than that for the middle prediction! 
# - We can write a short function to display the metrics.
# 
# <br></font>
# </div>

# In[ ]:


def show_metrics(metrics):
    """
    Make a boxplot of the metrics associated with prediction intervals
    
    :param metrics: dataframe of metrics produced from calculate error 
    :return fig: plotly figure
    """
    percent_in_bounds = metrics['in_bounds'].mean() * 100
    metrics_to_plot = metrics[[c for c in metrics if 'absolute_error' in c]]

    # Rename the columns
    metrics_to_plot.columns = [column.split('_')[-1].title() for column in metrics_to_plot]

    # Create a boxplot of the metrics
    fig = px.box(
        metrics_to_plot.melt(var_name="metric", value_name='Absolute Error'),
        x="metric",
        y="Absolute Error",
        color='metric',
        title=f"Error Metrics Boxplots    In Bounds = {percent_in_bounds:.2f}%",
        height=500,
        width=1000,
        points=False,
    )

    # Create new data with no legends
    d = []

    for trace in fig.data:
        # Remove legend for each trace
        trace['showlegend'] = False
        d.append(trace)

    # Make the plot look a little better
    fig.data = d
    fig['layout']['font'] = dict(size=20)
    return fig


# In[ ]:


iplot(show_metrics(metrics))


# In[ ]:


# Example plot subsetted to one WEEK -> YEAR-MONTH-DAY
fig = plot_intervals(predictions, mid=True, start="2017-03-01", stop="2017-03-08")
iplot(fig)


# # Creating a class for the process

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - To make this process repeatable, we can build our own estimator with a Scikit-Learn interface that fits and predicts all 3 models in one call each. 
# - This is a very simple class but can be extended based on your needs.
# 
# <br></font>
# </div>

# In[ ]:


class GradientBoostingPredictionIntervals(BaseEstimator):
    """
    Model that produces prediction intervals with a Scikit-Learn inteface
    
    :param lower_alpha: lower quantile for prediction, default=0.1
    :param upper_alpha: upper quantile for prediction, default=0.9
    :param **kwargs: additional keyword arguments for creating a GradientBoostingRegressor model
    """

    def __init__(self, lower_alpha=0.1, upper_alpha=0.9, **kwargs):
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha

        # Three separate models
        self.lower_model = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_alpha, **kwargs)
        
        self.mid_model = GradientBoostingRegressor(loss="ls", **kwargs)
        
        self.upper_model = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_alpha, **kwargs)
        self.predictions = None

    def fit(self, X, y):
        """
        Fit all three models
            
        :param X: train features
        :param y: train targets
        
        TODO: parallelize this code across processors
        """
        self.lower_model.fit(X_train, y_train)
        self.mid_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

    def predict(self, X, y):
        """
        Predict with all 3 models 
        
        :param X: test features
        :param y: test targets
        :return predictions: dataframe of predictions
        
        TODO: parallelize this code across processors
        """
        predictions = pd.DataFrame(y)
        predictions["lower"] = self.lower_model.predict(X)
        predictions["mid"] = self.mid_model.predict(X)
        predictions["upper"] = self.upper_model.predict(X)
        self.predictions = predictions

        return predictions

    def plot_intervals(self, mid=False, start=None, stop=None):
        """
        Plot the prediction intervals
        
        :param mid: boolean for whether to show the mid prediction
        :param start: optional parameter for subsetting start of predictions
        :param stop: optional parameter for subsetting end of predictions
    
        :return fig: plotly figure
        """

        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        fig = plot_intervals(predictions, mid=mid, start=start, stop=stop)
        return fig
    
    def calculate_and_show_errors(self):
        """
        Calculate and display the errors associated with a set of prediction intervals
        
        :return fig: plotly boxplot of absolute error metrics
        """
        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        calculate_error(self.predictions)
        fig = show_metrics(self.predictions)
        return fig


# In[ ]:


model = GradientBoostingPredictionIntervals(lower_alpha=0.1, upper_alpha=0.9, n_estimators=50, max_depth=3)

# Fit and make predictions
_ = model.fit(X_train, y_train)
predictions = model.predict(X_test, y_test)


# In[ ]:


metric_fig = model.calculate_and_show_errors()
iplot(metric_fig)


# In[ ]:


fig = model.plot_intervals(mid=True, start='2017-05-26', 
                           stop='2017-06-01')
iplot(fig)


# # Quantile loss explained

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# The quantile loss for a predicted is expressed as:
# - quantile loss = α∗(actual−predicted)if(actual−predicted)>0
# - quantile loss = (α−1)∗(actual−predicted)if(actual−predicted)<0
# 
# <br></font>
# </div>

# In[ ]:


def calculate_quantile_loss(quantile, actual, predicted):
    """
    Quantile loss for a given quantile and prediction
    """
    return np.maximum(quantile * (actual - predicted), (quantile - 1) * (actual - predicted))


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - When we graph the quantile loss versus the error, the weighting of the errors appears as the slope.
# - Let's walk through an example using lower quantile = 0.1, upper quantile = 0.9, and actual value = 10. 
# - There are four possibilities for the predictions:
# 
#     - Prediction = 15 with Quantile = 0.1. Actual < Predicted; Loss = (0.1 - 1) * (10 - 15) = 4.5
#     - Prediction = 5 with Quantile = 0.1. Actual > Predicted; Loss = 0.1 * (10 - 5) = 0.5
#     - Predicted = 15 with Quantile = 0.9. Actual < Predicted; Loss = (0.9 - 1) * (10 - 15) = 0.5
#     - Predicted = 5 with Quantile = 0.9. Actual < Predicted; Loss = 0.9 * (10 - 5) = 4.5
# 
# - For cases where the quantile > 0.5 we penalize low predictions more heavily. For cases where the quantile < 0.5 we 
# penalize high predictions more heavily.
# - If the quantile = 0.5, then the weighting is the same for both low and high predictions. For quantile == 0.5, we
# are predicting the median.
# 
# <br></font>
# </div>

# In[ ]:


def plot_quantile_loss(actual, prediction_list, quantile_list, plot_ls=False):
    """
    Shows the quantile loss associated with predictions at different quantiles.
    Figure shows the loss versus the error
    
    :param actual: array-like of actual values
    :param prediction_list: list of array-like predictions
    :param quantile_list: list of float quantiles corresponding to the predictions
    :param plot_ls: whether to plot the least squares loss
    
    :return fig: plotly figure
    """
    data = []

    # Iterate through each combination of prediction and quantile
    for predictions, quantile in zip(prediction_list, quantile_list):
        # Calculate the loss
        quantile_loss = calculate_quantile_loss(quantile, actual, predictions)
        
        errors = actual - predictions
        # Sort errors and loss by error
        idx = np.argsort(errors)
        errors = errors[idx]; quantile_loss = quantile_loss[idx]
    
        # Add data to plot
        data.append(go.Scatter(mode="lines", x=errors, y=quantile_loss, line=dict(width=4), name=f"{quantile} Quantile"))
        
    if plot_ls:
        loss = np.square(predictions - actual)
        errors = actual - predictions
        
        # Sort errors and loss by error
        idx = np.argsort(errors)
        errors = errors[idx]; loss = loss[idx]
    
        # Add data to plot
        data.append(go.Scatter(mode="lines", x=errors, y=loss, line=dict(width=4), name="Least Squares"))
        
    # Simple plot layout
    layout = go.Layout(
        title="Quantile Loss vs Error",
        yaxis=dict(title="Loss"),
        xaxis=dict(title="Error"),
        width=1000, height=500,
    )

    fig = go.Figure(data=data, layout=layout)
    fig['layout']['font'] = dict(size=18)
    return fig


# In[ ]:


# Make dummy predictions and actual values
predictions = np.arange(-2.1, 2.1, step=0.1)
actual = np.zeros(len(predictions))

# Create a plot showing the same predictions at different quantiles
fig = plot_quantile_loss(actual, [predictions, predictions, predictions], [0.1, 0.5, 0.9], False)
iplot(fig)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We can see how the quantile loss is asymmetric with a weighting (slope) equal to the quantile value or to 
# (quantile - 1) depending on if the error is positive or negative. 
# - With an error defined as (actual - predicted), for a quantile greater than 0.5, we penalize positive errors more and for a quantile less than 0.5, we penalize negative errors more. 
# - This drives the predictions with a higher quantile higher than the actual value, and predictions with a lower quantile lower than the actual value. The quantile loss is always positive.
# - This is a great reminder that the loss function of a machine learning method dictates what you are optimizing for!
# 
# <br></font>
# </div>

# In[ ]:


# Create plot with least squares loss as well
fig = plot_quantile_loss(actual, [predictions, predictions, predictions], [0.1, 0.5, 0.9], True)
iplot(fig)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - With the default loss function — least squares — the gradient boosting regressor is predicting the mean. 
# - The critical point to understand is that the least squares loss penalizes low and high errors equally. 
# - In contrast, the quantile loss penalizes errors based on the quantile and whether the error was positive (actual > predicted) or negative (actual < predicted). 
# - This allows the gradient boosting model to optimize not for the mean, but for percentiles.
# 
# <br></font>
# </div>

# In[ ]:


predictions = model.predictions.copy()

fig = plot_quantile_loss(
    predictions["actual"],
    [predictions["lower"], predictions["mid"], predictions["upper"]],
    [model.lower_alpha, 0.5, model.upper_alpha],
)
iplot(fig)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We can see the same weighting applied to the model's predictions. 
# - When the error is negative - meaning the actual value was less than the predicted value - and the quantile is less than 0.5, we weight the error by (quantile - 1)to penalize the high prediction. 
# - When the error is positive - meaning the actual value was greater than the predicted value - and the quantile is greater than 0.5, we weight the error by the quantile to penalize the low prediction.
# 
# <br></font>
# </div>

# # References

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# - https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed<br>
# - https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb<br>
# 
# <br></font>
# </div>

# In[ ]:




