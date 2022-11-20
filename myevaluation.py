import mysklearn.myutils as myutils
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       np.random.seed(random_state)
    
    if shuffle: 
        myutils.randomize_in_place(X, y)

    numInstances = len(X)
    if isinstance(test_size, float): # getting proportion, not set number
        test_size = math.ceil(numInstances * test_size)
    splitIndex = numInstances - test_size # index to split at
    
    return X[:splitIndex], X[splitIndex:], y[:splitIndex], y[splitIndex:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # get all indices for testing
    X_test_folds = []
    for i in range(n_splits):
        X_test_folds.append([])
    addIndex = 0
    for i in range(len(X)):
        X_test_folds[addIndex].append(i)
        addIndex = (addIndex + 1) % n_splits # determine bin to add to
    
    # add rest of indices to training set
    X_train_folds = []
    for i in range(n_splits):
        X_train_folds.append([])
    for j in range(n_splits):
        for i in range(len(X)):
            if i not in X_test_folds[j]:
                X_train_folds[j].append(i)
            

    return X_train_folds, X_test_folds # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # first, group by category name
    for i in range(len(X)):
        X[i].append(i)
        X[i].append(y[i])
    groupNames, groups = myutils.kFoldGroupBy(X)

    allGroups = []
    for group in groups:
        for row in group:
            allGroups.append(row)

    
    X_test_folds = []
    for i in range(n_splits):
        X_test_folds.append([])
    addIndex = 0
    # determine test set
    for i in range(len(allGroups)):
        X_test_folds[addIndex].append(allGroups[i][-1])
        addIndex = (addIndex + 1) % n_splits
    
    # add rest to training set
    X_train_folds = []
    for fold in X_test_folds:
        test = []
        for i in range(len(X)):
            if i not in fold:
                test.append(i)
        X_train_folds.append(test)

    return X_train_folds, X_test_folds # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    cMatrix = []
    for val in labels: # initializing matrix to all zeros
        row = []
        for i in range(len(labels)):
            row.append(0)
        cMatrix.append(row)

    for i in range(len(y_true)): # add one to correct index 
        rowIndex = labels.index(y_true[i])
        colIndex = labels.index(y_pred[i])
        cMatrix[rowIndex][colIndex] += 1

    return cMatrix
