import imp
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import sklearn.preprocessing as preproc
tab=['quarter','high', 'close', 'low', 'open', 'percent_change_next_weeks_price']

# read dataset from formatted file
dataset = pd.read_csv('dataset/dow_jones_index_formatted.data', header=0) # not worth to use if tf can do this better
filtered = dataset[['high','quarter', 'stock']]
grouped = filtered.groupby(filtered.quarter)

# split by quarters
train = grouped.get_group(1)
test = grouped.get_group(2)
grouped_stock = test.groupby(test.stock)
test = grouped_stock.get_group('INTC')

# remove useless columns
train.pop('quarter')
test.pop('quarter')
train.pop('stock')
test.pop('stock')


train = train.values
test = test.values

scaler = preproc.MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(train)
test_set_scaled = scaler.fit_transform(test)

train_x, train_y = [], []
test_x, test_y = [], []
SAMPLING = 3
i = SAMPLING
while i < 360:
    train_x.append(training_set_scaled[i-SAMPLING:i, 0])
    train_y.append(training_set_scaled[i,0])
#    test_x.append(test_set_scaled[i-SAMPLING:i, 0])
#    test_y.append(test_set_scaled[i, 0])
#    i += SAMPLING

i = SAMPLING
while i < 12:
    test_x.append(test_set_scaled[i-SAMPLING:i, 0])
    test_y.append(test_set_scaled[i, 0])
    i += SAMPLING

train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

model = keras.Sequential(
    [
        layers.SimpleRNN(256),
        layers.Dense(1)
    ]
)

model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam(),
    metrics=['accuracy'],
)
model.fit(
    train_x,
    train_y,
    batch_size=5,
    epochs=1
)
preds = model.predict(test_x)

from matplotlib import pyplot as plt

_, ax = plt.subplots()
ax.scatter([x for x in range(len(test_y))], test_y , c='red')
ax.scatter([x for x in range(len(preds))], preds, c='blue')
plt.show()