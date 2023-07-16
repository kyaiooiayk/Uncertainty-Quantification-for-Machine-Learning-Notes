#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Fully Probabilistic Bayesian CNN
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

tf.random.set_seed(84)
np.random.seed(84)


# # Basic Regression
# 

# In[2]:


x = tf.reshape((tf.range(0, 20, dtype = tf.float32)), [20, 1])
y = 2 * x + 1 + tf.random.normal(shape = tf.shape(x), mean = 0.5, stddev = 1.2)

plt.scatter(x, y)
plt.title('Data');


# In[3]:


def train_regression(model1, model2, model3, x, y):
    model1.compile(loss = 'mse', optimizer = 'adam')
    model1.fit(x, y, epochs = 1000, verbose = False)
    
    model2.compile(loss = 'mse', optimizer = 'adam')
    model2.fit(x, y, epochs = 1000, verbose = False)
    
    model3.compile(loss = 'mse', optimizer = 'adam')
    model3.fit(x, y, epochs = 1000, verbose = False)
    print('Model1: Prediction for 15:', model1.predict(tf.convert_to_tensor([15])))
    print('Model2: Prediction for 15:', model2.predict(tf.convert_to_tensor([15])))
    print('Model3: Prediction for 15:', model3.predict(tf.convert_to_tensor([15])))
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols=3, sharex = True, figsize = (20, 8),
                                        dpi = 128,
                                       sharey = True)

    ax1.scatter(x.numpy(), y.numpy(), label = 'Real Data', alpha = 0.4)
    ax1.scatter(x, model1.predict(x.numpy()), label = 'Model1 Prediction')
    ax1.legend(loc = 'upper left')

    ax2.scatter(x.numpy(), y.numpy(), label = 'Real Data', alpha = 0.4)
    ax2.scatter(x, model2.predict(x.numpy()), label = 'Model2 Prediction')
    ax2.legend(loc = 'upper left')

    ax3.scatter(x.numpy(), y.numpy(), label = 'Real Data', alpha = 0.4)
    ax3.scatter(x, model3.predict(x.numpy()), label = 'Model3 Prediction')    
    ax3.legend(loc = 'upper left')
    
    plt.show()


# In[4]:


# First Model
reg_model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation = 'relu',
                             kernel_initializer = 'ones'),
        tf.keras.layers.Dense(1)
    ])

# Second Model
reg_model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation = 'relu',
                             kernel_initializer = 'normal'),
        tf.keras.layers.Dense(1)
    ])

# Third Model
reg_model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation = 'relu',
                             kernel_initializer = 'he_uniform'),
        tf.keras.layers.Dense(1)
    ])

train_regression(reg_model_1,
                 reg_model_2,
                 reg_model_3, x, y)


# # Image Classification
# 
# <br>
# 
# ## Normal Model

# In[5]:


train_ds, test_ds = tfds.load('fashion_mnist', split = ['train', 'test'],
                             as_supervised = True)

train_ds = train_ds.batch(128).map(lambda x, y: (tf.cast(x / 255, tf.float32),
                                                tf.one_hot(y, 10))).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(128).map(lambda x, y: (tf.cast(x / 255, tf.float32),
                                                tf.one_hot(y, 10))).prefetch(tf.data.AUTOTUNE)


# In[6]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation = 'swish', 
                           input_shape = (28, 28, 1),
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.Conv2D(32, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(64, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(128, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.GlobalMaxPooling2D(),
    
    tf.keras.layers.Dense(128, activation = 'swish'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
             metrics = ['acc'])

model.summary()

model.fit(train_ds, epochs = 16, validation_data = test_ds)


# In[7]:


samples = []
labels = []

for x, y in test_ds.shuffle(buffer_size = 1024).take(1):
    samples.append(x.numpy())
    labels.append(np.argmax(y, axis = -1))
    
samples = np.squeeze(samples, axis = 0)
labels = np.squeeze(labels, axis = 0)
predictions = model.predict(samples)

assert samples.shape[0] == labels.shape[0]


# In[8]:


print(accuracy_score(labels, np.argmax(predictions, axis = -1)))


# In[9]:


def plot_predictions(predictions, labels):
    plt.figure(figsize = (20, 16))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.bar(range(10), predictions[i], color = 'red')
        plt.bar(range(10), predictions[i] - 0.03,
               color='white', linewidth=1, edgecolor='white')
        plt.title('Predicted %d // Actual: %d' % (np.argmax(predictions[i]),
                                                  labels[i]))
        plt.ylim([0, 1])
        plt.xticks(range(10))

    plt.tight_layout()
    plt.show()


# In[10]:


plot_predictions(predictions, labels)


# In[11]:


random_vector = np.random.random(size = (28, 28, 1))
random_vector.max(), random_vector.min()


# In[12]:


def random_noise_prediction(model = None, random_vector = None, ensemble = False):
    if ensemble:
        predictions = []
        for i in range(ensemble):
            predictions.append(model(random_vector[np.newaxis, :],
                        training = True).numpy().squeeze())
        predict_noise = np.mean(predictions, axis = 0)
    else:    
        predict_noise = model.predict(tf.expand_dims(random_vector,
                                              axis = 0)).squeeze()
    print({key:value for key, value in enumerate(predict_noise * 100)})                                             

    plt.figure(figsize = (8, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(10), predict_noise, color = 'red')
    plt.bar(range(10), predict_noise - 0.01,
       color='white', linewidth=1, edgecolor='white')
    plt.grid(False)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.subplot(1, 2, 2)
    plt.imshow(random_vector.squeeze(), cmap = plt.cm.gray)
    plt.show()


# In[13]:


random_noise_prediction(model, random_vector)


# In[14]:


for _ in range(5):
    predict_noise = model.predict(tf.expand_dims(random_vector,
                                             axis = 0)).squeeze()
    print('Softmax output for class 0:', predict_noise[0] * 100)


# ## Model with Dropouts

# In[15]:


model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation = 'swish', input_shape = (28, 28, 1),
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Conv2D(32, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(128, 3, activation = 'swish',
                           padding = 'same'),
    tf.keras.layers.GlobalMaxPooling2D(),
    
    tf.keras.layers.Dense(128, activation = 'swish'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
             metrics = ['acc'])

model2.summary()

model2.fit(train_ds, epochs = 16, validation_data = test_ds)


# In[16]:


for _ in range(4):
    predict_noise = model2(tf.expand_dims(random_vector,axis = 0),
                        training = True).numpy().squeeze()
    print('Softmax output for class 0:', predict_noise[0] * 100)


# In[17]:


random_noise_prediction(model2, random_vector, ensemble = 100)


# In[18]:


predictions = []
for _ in range(1000):
    predicted_ensemble = model2(samples,
                        training = True).numpy().squeeze()
    predictions.append(predicted_ensemble)
predictions = np.array(predictions)
print('Predictions shape:', predictions.shape)
predictions_median = np.median(predictions, axis = 0)


# In[19]:


predictions_median.shape


# In[20]:


plot_predictions(predictions_median, labels)


# In[21]:


print(accuracy_score(labels, np.argmax(predictions_median, axis = -1)))


# ## Samples with Confidence Intervals

# In[22]:


def plot_with_percentiles(prediction_normal,
                          prediction_dropout_ensemble):
    dropout_median_predictions = np.median(prediction_dropout_ensemble,
                                          axis = 0) 
    ensemble_pct_97_5 = np.array([np.percentile(prediction_dropout_ensemble[:, j], 
                                                97.5) for j in range(10)]).squeeze()
    ensemble_pct_2_5 = np.array([np.percentile(prediction_dropout_ensemble[:, j], 
                                               2.5) for j in range(10)]).squeeze()

    fig, (ax1, ax2, ax3) = plt.subplots(figsize = (14, 5),
                                        nrows = 1, ncols = 3, sharex = True, sharey = True)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    
    ax1.bar(range(10), prediction_normal.squeeze(), color = 'red')
    ax1.bar(range(10), prediction_normal.squeeze() - 0.03, color='white',
          linewidth=1, edgecolor='white')
    ax1.set_title('Softmax Output')
    
    ax2.bar(range(10), dropout_median_predictions, color = 'purple')
    ax2.bar(range(10), dropout_median_predictions.squeeze() - 0.03, color='white',
            linewidth=1, edgecolor='white')
    ax2.set_title('Dropout Ensemble Median')
    
    ax3.bar(range(10), ensemble_pct_97_5, color='green')
    ax3.bar(range(10), ensemble_pct_2_5 - 0.03, color='white',
          linewidth=1, edgecolor='white')
    ax3.set_title('Dropout Ensemble 95% CI')
    
    plt.show()


# In[23]:


def get_normal_and_ensemble_preds(samples = samples, model_normal = model,
                                 model_ensemble = model2,
                                 class_idx = 0, ensemble_size = 100):
    selected_sample = samples[class_idx]
    
    prediction_normal = model_normal.predict(selected_sample[np.newaxis, :])

    prediction_dropout = []
    for i in range(ensemble_size):
        prediction_i = model_ensemble(selected_sample[np.newaxis, :],
                            training = True).numpy().squeeze()
        prediction_dropout.append(prediction_i) 
    prediction_dropout = np.array(prediction_dropout)
    return prediction_normal, prediction_dropout


# In[24]:


for idx in [0, 10, 20, 25, 30, 36, 45, 50, 56, 64, 80, 92, 127]:
    prediction_normal, prediction_dropout_ensemble = get_normal_and_ensemble_preds(class_idx = idx, 
                                                                                   ensemble_size = 1000)
    plot_with_percentiles(
            prediction_normal,
            prediction_dropout_ensemble
    )
    del prediction_normal, prediction_dropout_ensemble


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/Frightera/Medium_Notebooks_English/blob/main/Brief%20Introduction%20to%20Uncertainty/Brief%20Introduction%20to%20Uncertainty%20-%20Medium.ipynb
# - https://towardsdatascience.com/uncertainty-in-deep-learning-brief-introduction-1f9a5de3ae04
# 
# </font>
# </div>

# # Requirements
# <hr style = "border:2px solid black" ></hr>

# In[25]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')


# In[ ]:




