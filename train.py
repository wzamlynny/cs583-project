
from model import *
from data import *

from keras.utils import to_categorical

import numpy as np

# Get the data
X, Y = load_train_data()

#X = X[0] # For non-image training, just use the labels

# For single image training, just use the first image
for i in range(len(X[1])):
    if len(X[1][i]) > 0:
        X[1][i] = X[1][i][0]
    else:
        X[1][i] = np.zeros((64,64,3))

X[1] = np.array(X[1])

# Shuffle the data
idxs = list(range(len(Y)))
X = list(map(lambda X: X[idxs], X))
Y = Y[idxs]

# Display the distribution of each possible output
hist = np.histogram(Y, bins=[0,1,2,3,4,5])
print(hist)
for i in range(5):
    print('Num w/ label', i, ':', hist[0][i])
print('Total:', len(Y))

# One hot encode the output
Y = to_categorical(Y)

# Validation split
s = int(0.2 * len(X))
X_train = X[:-s]
Y_train = Y[:-s]
X_valid = X[-s:]
Y_valid = Y[-s:]

print('Training points:', len(X_train))
print('Validation points:', len(X_valid))
print('Total points:', len(X))

# Build a model
#clf = ImageFreeModel((X_train, Y_train), (X_valid, Y_valid))
clf = ConvModel((X_train, Y_train), (X_valid, Y_valid))

# Build the model
clf.compile()

# Fit to the data
clf.train(epochs=32)

