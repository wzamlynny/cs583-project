
from model import *
from data import *

from keras.utils import to_categorical

import numpy as np

# Get the data
X, Y = load_train_data()

#X = X[0] # For non-image training, just use the labels

def shuffle(X, Y):
    idxs = list(range(len(Y)))
    
    if isinstance(X, list):
        for i in range(len(X)):
            X[i] = X[i][idxs]
    else:
        X = X[idxs]

    Y = Y[idxs]

def split(X, Y, s=0.2):
    s = int(s * len(Y))
    
    if isinstance(X, list):
        X_train = [x[:-s] for x in X]
        X_test = [x[-s:] for x in X]
    else:
        X_train = X[:-s]
        X_test = X[-s:]

    Y_train = Y[:-s]
    Y_test = Y[-s:]

    return (X_train, Y_train), (X_test, Y_test)


def convert_for_all(X, Y):
    # For single image training, make a datapoint for each image or the default zero image
    Xs = [[], []]
    Ys = []
    for i in range(len(X[1])):
        Xs[0].append(X[0][i])
        if len(X[1][i]) == 0:
            Xs[1].append(np.zeros((64, 64, 3)))
            Ys.append(Y[i])
        else:
            # Make a datapoint for all images. We assume equal relevance of images
            for img in X[1][i]:
                Xs[0].append(X[0][i])
                Xs[1].append(img)
                Ys.append(Y[i])

    X = list(map(np.array, Xs))
    Y = np.array(Ys)

    return X, Y

def convert_for_single_axis(X, Y, ax=0):
    return X[ax], Y

def train_model(mdl, X, Y, epochs=32):
    # Shuffle the data
    shuffle(X, Y)

    # One hot encode the output
    Y = to_categorical(Y)

    # Validation split
    (X_train, Y_train), (X_valid, Y_valid) = split(X, Y)

    print('Training points:', len(Y_train))
    print('Validation points:', len(Y_valid))
    print('Total points:', len(Y))

    clf = mdl((X_train, Y_train), (X_valid, Y_valid))

    # Build the model
    clf.compile()

    # Fit to the data
    clf.train(epochs=epochs)

    return clf


# Attribute model data
X_attr, Y_attr = convert_for_single_axis(X, Y, ax=0)
# Image-free model
attr_clf = ImageFreeModel
# Train the model
attr_clf = train_model(attr_clf, X_attr, Y_attr)

# Create inputs for convolutional model
X_conv, Y_conv = convert_for_all(X, Y)
X_conv = X_conv[1]
print(X_conv.shape)
# Build a model
conv_clf = SingleImageModel
# Train the model
conv_clf = train_model(conv_clf, X_conv, Y_conv)

# Create inputs for convolutional model
X_conv, Y_conv = convert_for_all(X, Y)
# Build a model
conv_clf = lambda tr, tst: ConvModel([conv_clf, attr_clf], tr, tst)
# Train the model
conv_clf = train_model(conv_clf, X_conv, Y_conv)

