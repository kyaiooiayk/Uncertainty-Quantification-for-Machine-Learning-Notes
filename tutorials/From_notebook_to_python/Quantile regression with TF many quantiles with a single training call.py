#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Quantile-regression-with-one-training-call" data-toc-modified-id="Quantile-regression-with-one-training-call-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Quantile regression with one training call</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Import-the-dataset" data-toc-modified-id="Import-the-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import the dataset</a></span></li><li><span><a href="#Build-the-model" data-toc-modified-id="Build-the-model-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Build the model</a></span></li><li><span><a href="#Out-of-sample-predictions" data-toc-modified-id="Out-of-sample-predictions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Out-of-sample predictions</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Quantile regression with TF many quantiles with a single training call
# 
# **Reference [code]:** https://github.com/strongio/quantile-regression-tensorflow/blob/master/Quantile%20Loss.ipynb<br>    
# 
# <br></font>
# </div>

# # Quantile regression with one training call

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - There should be a way to compute the quantile in one go without having to re-run the model twice (lower & upper models).
# - This does seems to be an easy thing to do in KERAS as exaplined in the article.
# - However you can do it in **Tensor Flow** as shown in here: https://github.com/strongio/quantile-regression-tensorflow/blob/master/Quantile%20Loss.ipynb
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 17, 8
rcParams['font.size'] = 20

# https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# # Import the dataset

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Standardise the inputs and outputs so that it is easier to train.
# - So apparently the guy is standardisining the input for regression, something that I was wondering about but I have not found something to back it up.
# 
# <br></font>
# </div>

# In[2]:


mcycle = pd.read_csv('./motorcycleAccident.txt', delimiter = '\t')


# In[3]:


mcycle.times = (mcycle.times - mcycle.times.mean())/mcycle.times.std()
mcycle.accel = (mcycle.accel - mcycle.accel.mean())/mcycle.accel.std()


# In[4]:


# Reshape to input format for network
times = np.expand_dims(mcycle.times.values, 1)
accel = np.expand_dims(mcycle.accel.values, 1)


# # Build the model

# In[5]:


# Create network
class q_model:
    def __init__(self,
                 sess,
                 quantiles,
                 in_shape=1,
                 out_shape=1,
                 batch_size=32):

        self.sess = sess

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch_size = batch_size

        self.outputs = []
        self.losses = []
        self.loss_history = []

        self.build_model()

    def build_model(self, scope='q_model', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.in_shape))
            self.y = tf.placeholder(tf.float32, shape=(None, self.out_shape))

            self.layer0 = tf.layers.dense(self.x,
                                          units=32,
                                          activation=tf.nn.relu)
            self.layer1 = tf.layers.dense(self.layer0,
                                          units=32,
                                          activation=tf.nn.relu)

            # Create outputs and losses for all quantiles
            for i in range(self.num_quantiles):
                q = self.quantiles[i]

                # Get output layers
                output = tf.layers.dense(
                    self.layer1, 1, name="{}_q{}".format(i, int(q*100)))
                self.outputs.append(output)

                # Create losses

                error = tf.subtract(self.y, output)
                loss = tf.reduce_mean(tf.maximum(
                    q*error, (q-1)*error), axis=-1)

                self.losses.append(loss)

            # Create combined loss
            self.combined_loss = tf.reduce_mean(tf.add_n(self.losses))
            self.train_step = tf.train.AdamOptimizer().minimize(self.combined_loss)

    def fit(self, x, y, epochs=100):
        for epoch in range(epochs):
            epoch_losses = []
            for idx in range(0, x.shape[0], self.batch_size):
                batch_x = x[idx: min(idx + self.batch_size, x.shape[0]), :]
                batch_y = y[idx: min(idx + self.batch_size, y.shape[0]), :]

                feed_dict = {self.x: batch_x,
                             self.y: batch_y}

                _, c_loss = self.sess.run(
                    [self.train_step, self.combined_loss], feed_dict)
                epoch_losses.append(c_loss)

            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)
            if epoch % 100 == 0:
                print("Epoch {}: {}".format(epoch, epoch_loss))

    def predict(self, x):
        # Run model to get outputs
        feed_dict = {self.x: x}
        predictions = sess.run(self.outputs, feed_dict)

        return predictions


# In[6]:


# Initialize session
sess = tf.Session()

# Instantiate model
quantiles = [.1, .5, .9]
model = q_model(sess, quantiles, batch_size=32)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[7]:


# Run training
epochs = 2000
model.fit(times, accel, epochs)


# In[8]:


# Generate the range of data we'd like to predict
test_times = np.expand_dims(np.linspace(times.min(),times.max(),200), 1)
predictions = model.predict(test_times)

plt.scatter(times, accel)
for i, prediction in enumerate(predictions):
    plt.plot(test_times, prediction, label='{}th Quantile'.format(int(model.quantiles[i]*100)))
    
plt.legend()
plt.show()


# # Out-of-sample predictions
# <hr style="border:2px solid black"> </hr>

# In[12]:


# Generate the range of data we'd like to predict
test_times = np.expand_dims(np.linspace(0, 4, 200), 1)
predictions = model.predict(test_times)

plt.scatter(times, accel)
for i, prediction in enumerate(predictions):
    plt.plot(test_times, prediction, label='{}th Quantile'.format(
        int(model.quantiles[i]*100)))

plt.legend()
plt.show()


# # Conclusion

# <div class="alert alert-block alert-danger">
# <font color=black><br>
# 
# - TF over Keras if you want to save time. 
# 
# <br></font>
# </div>

# In[ ]:




