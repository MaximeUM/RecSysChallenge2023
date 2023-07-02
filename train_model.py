#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

import os
import glob
import math


# In[ ]:


tf.random.set_seed(2023)
import random
random.seed(2023)
np.random.seed(2023)


# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


tf.config.list_physical_devices('GPU')


# ### Load data

# In[ ]:


path = "data/train/"
all_files = glob.glob(os.path.join(path, "*.csv"))

data = pd.concat((pd.read_csv(f, sep="\t") for f in all_files), ignore_index=True)


# In[ ]:


test_data = pd.read_csv("data/test/000000000000.csv", sep="\t")


# ### Preprocess data

# In[ ]:


cat = data.iloc[:,2:33]
del cat["f_7"] #Only one value --> useless
bin = data.iloc[:,33:42]
num = data.iloc[:,42:80]
labels = data.iloc[:,80:82]

cat_test = test_data.iloc[:,2:33]
del cat_test["f_7"] #Only one value --> useless
bin_test = test_data.iloc[:,33:42]
num_test = test_data.iloc[:,42:80]


# In[ ]:


# Categorical variables
cat_selected = cat.to_numpy()
cat_test_selected = cat_test.to_numpy()

# Numerical variables : estimate missing values and normalize 
imputer = IterativeImputer(max_iter=10, random_state=0)
num_selected = imputer.fit_transform(num)
scaler = MinMaxScaler()
num_selected = scaler.fit_transform(num_selected)

num_test_selected = scaler.transform(imputer.transform(num_test))

# Binary variables
bin_selected = bin.to_numpy()
bin_test_selected = bin_test.to_numpy()

# Output variables
y = labels
# y_is_clicked = y.iloc[:,0]
y_is_installed = y.iloc[:,1]


# In[ ]:


for col_ind in range(cat_selected.shape[1]):

    unique_values = np.unique(cat_selected[:, col_ind][~np.isnan(cat_selected[:,col_ind])]).astype(int)
    # test_unique_values = np.unique(cat_test_selected[:, col_ind]).astype(int)

    # Make categorical variables from 1 to n (n corresponding to the number of unique values for the corresponding categorical feature)
    replacement_dict = dict()
    for index, val in enumerate(unique_values):
        index+=1
        replacement_dict[val] = index

    # Process training data (categorical)
    for line_ind in range(len(cat_selected[:,col_ind])):
        if math.isnan(cat_selected[line_ind, col_ind]): # 0 used for missing values
            cat_selected[line_ind, col_ind] = 0
        else:
            cat_selected[line_ind, col_ind] = replacement_dict[int(cat_selected[line_ind, col_ind])] # Use the new value (from 1 to n)
    
    # Process test data (categorical)
    for line_ind in range(len(cat_test_selected[:,col_ind])):
        try:
            if math.isnan(cat_test_selected[line_ind, col_ind]):
                cat_test_selected[line_ind, col_ind] = 0 # 0 used for missing values
            else:
                cat_test_selected[line_ind, col_ind] = replacement_dict[int(cat_test_selected[line_ind, col_ind])] # Use the new value (from 1 to n)
        except KeyError:
            cat_test_selected[line_ind, col_ind] = 0 # If the value was not in the training data, treat as a missing value (because we can't train on it)

cat_selected = cat_selected.astype(int)
cat_test_selected = cat_test_selected.astype(int)


# In[ ]:


# Compute bias to help the model with imbalanced dataset
neg, pos = np.bincount(y_is_installed) 
total = neg + pos 
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format( total, pos, 100 * pos / total)) 
initial_bias = np.log([pos/neg]) 
print(initial_bias)


# In[ ]:


embed_size = 256

# Inputs
cat_input_layer = layers.Input(shape=(cat_selected.shape[1],), dtype=tf.int64)
bin_input_layer = layers.Input(shape=(bin_selected.shape[1],), dtype=tf.int64)
num_input_layer = layers.Input(shape=(num_selected.shape[1],), dtype=tf.float64)

embedding_layers = []
for i in range(cat_selected.shape[1]):
    num_values = len(set(cat_selected[:,i]))
    if num_values <= embed_size:
        embedding_layers.append(layers.Embedding(input_dim=num_values+1, output_dim=num_values, input_length=1, mask_zero=True)(cat_input_layer[:,i]))
    else:
        embedding_layers.append(layers.Embedding(input_dim=num_values+1, output_dim=embed_size, input_length=1, mask_zero=True)(cat_input_layer[:,i]))

bin_dense_layer = layers.Dense(64, activation='relu')(bin_input_layer)
num_dense_layer = layers.Dense(64, activation='relu')(num_input_layer)

# Concat all inputs
concatted = tf.keras.layers.Concatenate()([bin_dense_layer, num_dense_layer, *embedding_layers])

# Hidden layers
hidden_layer_1 = layers.Dense(500, activation='relu')(concatted)
hidden_layer_2 = layers.Dense(250, activation='relu')(hidden_layer_1)
hidden_layer_3 = layers.Dense(50, activation='relu')(hidden_layer_2)
hidden_layer_4 = layers.Dense(100, activation='relu')(hidden_layer_3)
hidden_layer_5 = layers.Dense(40, activation='relu')(hidden_layer_4)

# Outputs
output_bias = tf.keras.initializers.Constant(initial_bias)

# output_1 = layers.Dense(1, activation='sigmoid', name="is_clicked", bias_initializer=output_bias)(hidden_layer_5) # If we want to predict "is_clicked", use this output (give two outputs to the model instead of one).
output_2 = layers.Dense(1, activation="sigmoid", name="is_installed", bias_initializer=output_bias)(hidden_layer_5)

# Create model
model = keras.Model(inputs=[cat_input_layer, bin_input_layer, num_input_layer], outputs=[output_2])
# model = keras.Model(inputs=[cat_input_layer, bin_input_layer, num_input_layer], outputs=[output_1, output_2]) # If we want to predict "is_clicked" and "is_installed"

# Compile model
batch_size = 5000 
learning_rate=0.001
optimizer = keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])   


# In[ ]:


# Validation dataset is better to know when to stop training (to do before using the model in production)
for i in range(3): # First three training epochs will give good results, after that we would overfit (based on previous tests)
    # model.fit(x_norm,y_is_installed, epochs=40, batch_size=5000)
    model.fit((cat_selected, bin_selected, num_selected), y_is_installed, epochs = 1, batch_size=batch_size)
    model.save("models/mv_iterative_embeds/" + str(i))
    y_test = model.predict((cat_test_selected, bin_test_selected, num_test_selected), batch_size=batch_size) 

    results = pd.DataFrame(test_data["f_0"].copy())
    results["row_id"] = results["f_0"]
    del results["f_0"]

    results["is_clicked"] = 0 # Not predicted here, but can be predicted if the model is configured with two outputs
    results["is_installed"] = y_test

    save_path="results/mv_iterative_embeds/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results.to_csv(os.path.join(save_path, str(i) + ".csv"), index=False, header=True, sep="\t")

