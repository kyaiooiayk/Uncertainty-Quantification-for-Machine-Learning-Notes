#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Tensor flow and Monte Carlo method for PIs
# 
# **Reference [code]:** https://github.com/ceshine/quantile-regression-tensorflow/blob/master/notebooks/01-sklearn-example.ipynb<br>
# **Reference [article]:** https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629<br>
# 
# <br></font>
# </div>

# # Import modules

# In[11]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 17, 8
rcParams['font.size'] = 20

# https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)


# # Create the dataset

# In[4]:


def f(x):
    """The function to predict."""
    return x * np.sin(x)

#  First the noiseless case
X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
X = X.astype(np.float32)

# Observations
y = f(X).ravel()

dy = 1.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise
y = y.astype(np.float32)

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
xx = xx.astype(np.float32)

X.shape, y.shape, xx.shape


# # Build the model

# In[5]:


# Create network
class q_model:
    def __init__(self,
                 sess,
                 quantiles,
                 in_shape=1,
                 out_shape=1,
                 batch_size=32,
                 dropout=0.5):

        self.sess = sess

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch_size = batch_size
        self.dropout = dropout

        self.outputs = []
        self.losses = []
        self.loss_history = []
        self.optim = tf.train.AdamOptimizer()

        self.build_model()

    def build_model(self, scope='q_model', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.in_shape))
            self.y = tf.placeholder(tf.float32, shape=(None, self.out_shape))
            self.is_training = tf.placeholder(tf.bool)

            self.layer0 = tf.layers.dense(self.x,
                                          units=64,
                                          activation=tf.nn.relu)
            # self.layer0_dropped = tf.layers.dropout(tf.layers.batch_normalization(
            #     self.layer0), self.dropout, training=self.is_training)
            self.layer0_dropped = tf.layers.dropout(
                self.layer0, self.dropout, training=self.is_training)
            self.layer1 = tf.layers.dense(self.layer0,
                                          units=64,
                                          activation=tf.nn.relu)
            # self.layer1_dropped = tf.layers.dropout(tf.layers.batch_normalization(
            #     self.layer1), self.dropout, training=self.is_training)
            self.layer1_dropped = tf.layers.dropout(
                self.layer1, self.dropout, training=self.is_training)
            # Create outputs and losses for all quantiles
            for i in range(self.num_quantiles):
                q = self.quantiles[i]

                # Get output layers
                output = tf.layers.dense(
                    self.layer1_dropped, 1, name="{}_q{}".format(i, int(q*100)))
                self.outputs.append(output)

                # Create losses
                error = tf.subtract(self.y, output)
                loss = tf.reduce_mean(tf.maximum(
                    q*error, (q-1)*error), axis=-1)
                self.losses.append(loss)

            # Create combined loss with weight loss
            # self.combined_loss = tf.add(
            #     tf.reduce_mean(tf.add_n(self.losses)),
            #     1e-4 * tf.reduce_sum(tf.stack(
            #       [tf.nn.l2_loss(i) for i in tf.trainable_variables()]
            #     ))
            # )
            # Create combined loss
            self.combined_loss = tf.reduce_mean(tf.add_n(self.losses))
            self.train_step = self.optim.minimize(self.combined_loss)

    def fit(self, x, y, epochs=100):
        for epoch in range(epochs):
            epoch_losses = []
            shuffle_idx = np.arange(x.shape[0])
            np.random.shuffle(shuffle_idx)
            x = x[shuffle_idx]
            y = y[shuffle_idx]
            for idx in range(0, x.shape[0], self.batch_size):
                batch_x = x[idx: min(idx + self.batch_size, x.shape[0]), :]
                batch_y = y[idx: min(idx + self.batch_size, y.shape[0]), :]

                feed_dict = {
                    self.x: batch_x,
                    self.y: batch_y,
                    self.is_training: True
                }

                _, c_loss = self.sess.run(
                    [self.train_step, self.combined_loss], feed_dict)
                epoch_losses.append(c_loss)

            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)
            if epoch % 500 == 0:
                print("Epoch {}: {}".format(epoch, epoch_loss))

    def predict(self, x, is_training=False):
        # Run model to get outputs
        feed_dict = {self.x: x, self.is_training: is_training}
        predictions = sess.run(self.outputs, feed_dict)

        return [x[:, 0] for x in predictions]


# # Train the model

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Look at the **dropout** argument
# 
# <br></font>
# </div>

# In[7]:


# Initialize session
sess = tf.Session()

# Instantiate model
quantiles = [.05, .5, .95]
model = q_model(sess, quantiles, batch_size=10, dropout=0.25)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[8]:


# Run training
epochs = 10000
model.fit(X, y[:, np.newaxis], epochs)


# In[9]:


# Make the prediction on the meshed x-axis
y_lower, y_pred, y_upper = model.predict(xx)


# In[12]:


# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(xx, y_pred, 'r-', label=u'Prediction')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# # Some checks

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - As you can see the results are acceptable.
# - Maybe some tuning would help.
# 
# <br></font>
# </div>

# In[15]:


predictions = model.predict(X)
np.mean(predictions[0]), np.mean(predictions[1]), np.mean(predictions[2])


# In[16]:


in_the_range = np.sum((y >= predictions[0]) & (y <= predictions[2]))
print("Percentage in the range (expecting 90%):", in_the_range / len(y) * 100)


# In[17]:


out_of_the_range = np.sum((y < predictions[0]) | (y > predictions[2]))
print("Percentage out of the range (expecting 10%):", out_of_the_range / len(y)  * 100)


# # Montel Carlo dropout

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - MC dropout oerforms dropout in test/prediction time to approximate sampling from the posterior distribution. 
# - The we can find the credible interval for a quantile. 
# 
# <br></font>
# </div>

# In[19]:


K = 5000
tmp = np.zeros((K, xx.shape[0])).astype("float32")
for k in range(K):
    _, preds, _ = model.predict(xx, is_training=True)
    tmp[k] = preds
y_lower, y_pred, y_upper = np.percentile(tmp, (5, 50, 95), axis=0)


# In[20]:


y_lower[1], y_pred[1], y_upper[1]


# In[21]:


# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(xx, y_pred, 'r-', label=u'Prediction')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% credible interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


# # Some checks

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - As written in the article the authors **is not sure** how to make this method works optimally.
# 
# <br></font>
# </div>

# In[22]:


tmp = np.zeros((K, X.shape[0])).astype("float32")
for k in range(K):
    _, preds, _ = model.predict(X, is_training=True)
    tmp[k] = preds
predictions = np.percentile(tmp, (5, 50, 95), axis=0)
np.mean(predictions[0]), np.mean(predictions[1]), np.mean(predictions[2])


# In[23]:


in_the_range = np.sum((y >= predictions[0]) & (y <= predictions[2]))
print("Percentage in the range:", in_the_range / len(y) * 100)


# In[24]:


out_of_the_range = np.sum((y < predictions[0]) | (y > predictions[2]))
print("Percentage out of the range:", out_of_the_range / len(y)  * 100)


# In[ ]:




