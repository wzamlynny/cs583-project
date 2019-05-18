# You must compare your fancy methods with simple baselines,
# e.g., random guess, all-positive, all-negative, simple 
# linear models, and beat the baselines. The evaluation metric
# must be the one required by the competition.
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LinearRegression
from sklearn import tree

# Each model should take in the training_x, training_y, test_x, and test_y
# the other ones take in as such: ConvModel((X_train, Y_train), (X_valid, Y_valid))


def random_guess(X):
    return np.random.randint(0,5, len(X))

def all_0(X):
    return np.zeros(len(X))

def all_1(X):
    return np.ones(len(X))

def all_2(X):
    return np.ones(len(X))*2

def all_3(X):
    return np.ones(len(X))*3

def all_4(X):
    return np.ones(len(X))*4

# Should this output 3 accuracies?
# TODO Should this just run on entire dataset with one score?
def run_random_guess(train, valid=None, test=None):
    print("Running random guess...")
    # Run "Training"
    train_x, train_y = train
    train_pred = random_guess(train_x)
    train_score = accuracy_score(train_y, train_pred)
    print("'Training' accuracy: {}".format(train_score))
    train_kappa = cohen_kappa_score(train_y, train_pred, weights='quadratic')
    print("'Training' kappa: {}".format(train_kappa))

    if valid is not None:
        # Run "Validation"
        valid_x, valid_y = valid
        valid_pred = random_guess(valid_x)
        valid_score = accuracy_score(valid_y, valid_pred)
        print("'Validation' accuracy: {}".format(valid_score))
        valid_kappa = cohen_kappa_score(valid_y, valid_pred, weights='quadratic')
        print("'Validation' kappa: {}".format(valid_kappa))
    
    if test is not None:
        # Return results from "Testing" for Kaggle output
        return random_guess(test)

def run_all_n_guess(train, valid=None, test=None):
    ret = np.zeros((5, len(test)))
    for i in range(0,5):
        print("Running all '{}' guess...".format(i))
        # Run "Training"
        train_x, train_y = train
        train_pred = eval("all_{}(train_x)".format(i))
        train_score = accuracy_score(train_y, train_pred)
        print("'Training' accuracy: {}".format(train_score))
        train_kappa = cohen_kappa_score(train_y, train_pred, weights='quadratic')
        print("'Training' kappa: {}".format(train_kappa))

        if valid is not None:
            # Run "Validation"
            valid_x, valid_y = valid
            valid_pred = eval("all_{}(valid_x)".format(i))
            valid_score = accuracy_score(valid_y, valid_pred)
            print("'Validation' accuracy: {}".format(valid_score))
            valid_kappa = cohen_kappa_score(valid_y, valid_pred, weights='quadratic')
            print("'Validation' kappa: {}".format(valid_kappa))  

        if test is not None:
            # Return results from "Testing" for Kaggle output
            ret[i] = eval("all_{}(test)".format(i))
    if test is not None:
        return ret

def linear_regression(train, valid, test):
    # Linear regression with rounding to the n
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x = test

    model = LinearRegression().fit(train_x, train_y)
    train_pred = np.around(model.predict(train_x))
    train_score = accuracy_score(train_y, train_pred)
    print("Training accuracy: {}".format(train_score))
    train_kappa = cohen_kappa_score(train_y, train_pred, weights='quadratic')
    print("Training kappa: {}".format(train_kappa))

    valid_pred = np.around(model.predict(valid_x))
    valid_score = accuracy_score(valid_y, valid_pred)
    print("Validation accuracy: {}".format(valid_score))
    valid_kappa = cohen_kappa_score(valid_y, valid_pred, weights='quadratic')
    print("Validation kappa: {}".format(valid_kappa)) 

    test_pred = np.around(model.predict(test_x))
    return test_pred

def decision_tree(train, valid, test):
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x = test

    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(train_x, train_y)

    train_pred = clf.predict(train_x)
    train_score = accuracy_score(train_y, train_pred)
    print("Training accuracy: {}".format(train_score))
    train_kappa = cohen_kappa_score(train_y, train_pred, weights='quadratic')
    print("Training kappa: {}".format(train_kappa))

    valid_pred = clf.predict(valid_x)
    valid_score = accuracy_score(valid_y, valid_pred)
    print("Validation accuracy: {}".format(valid_score))
    valid_kappa = cohen_kappa_score(valid_y, valid_pred, weights='quadratic')
    print("Validation kappa: {}".format(valid_kappa)) 

    test_pred = clf.predict(test_x)
    return clf, test_pred