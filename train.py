
from model import *
from data import *

from keras.utils import to_categorical

import numpy as np

# Get the data
X, Y = load_train_data()

#X = X[0] # For non-image training, just use the labels

# For single image training, make a datapoint for each image or the default zero image
Xs = [[], []]
for i in range(len(X[1])):
    Xs[0].append(X[0][i])
    if len(X[1][i]) == 0:
        Xs[1].append(np.zeros((64, 64, 3)))
    else:
        # Make a datapoint for all images. We assume equal relevance of images.
        for img in X[1][i]:
            Xs[0].append(X[0][i])
            Xs[1].append(img)

# Convert all to numpy
X = list(map(np.array, Xs))

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
s = int(0.2 * len(Y))
X_train = [x[:-s] for x in X]
Y_train = Y[:-s]
X_valid = [x[-s:] for x in X]
Y_valid = Y[-s:]

print('Training points:', len(Y_train))
print('Validation points:', len(Y_valid))
print('Total points:', len(Y))

# Build a model
#clf = ImageFreeModel((X_train, Y_train), (X_valid, Y_valid))
clf = ConvModel((X_train, Y_train), (X_valid, Y_valid))

# Build the model
clf.compile()

# Fit to the data
clf.train(epochs=32)

