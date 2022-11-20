import mysklearn.myutils as myutils
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x = myutils.convert_2D_to_1D(X_train)
        y = myutils.convert_2D_to_1D(y_train)
        slope, intercept = myutils.compute_slope_intercept(x, y_train)
        self.slope = slope 
        self.intercept = intercept

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []

        xOneDimension = myutils.convert_2D_to_1D(X_test)
        for x in xOneDimension:
            yVal = self.slope * x + self.intercept 
            predictions.append(yVal)
        return predictions


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        allDist = []
        allIndices = []

        for instance in X_test:
            dist, ind = myutils.getKNeighbors(instance, self.X_train, self.n_neighbors)
            allDist.append(dist)
            allIndices.append(ind)


        return allDist, allIndices 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, indices = self.kneighbors(X_test)
        predictions = []
        for val in indices:
            predictHolder = []
            for index in val:
                predictHolder.append(self.y_train[index])
            classification = myutils.getClassification(predictHolder)
            predictions.append(classification)
        return predictions

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        numCols = len(X_train[0])
        classNames, self.priors = myutils.getPriors(y_train)
        totalClass = [0 for val in classNames]
        for val in y_train:
            index = classNames.index(val)
            totalClass[index] += 1
        
        postDict = {}
        for i in range(numCols):
            postDict[i] = {}
            attVals = myutils.getUniqueIdentifiers(X_train, i) 
            for j, vals in enumerate(attVals):
                postDict[i][vals] = {}
  
                for name in classNames:
                    postDict[i][vals][name] = 0.0
     
        for i in range(numCols):
            for j, row in enumerate(X_train):

                postDict[i][row[i]][y_train[j]] += 1
     
        for i in range(numCols):
            attVals = myutils.getUniqueIdentifiers(X_train, i) 
            for val in attVals:
                for k, name in enumerate(classNames):
                    postDict[i][val][name] /= totalClass[classNames.index(name)]
     
        self.posteriors = postDict

    def getSinglePrediction(self, instance):
        numAtts = len(instance)
        classes = sorted(list(set(self.y_train)))
        
        allProbs = []
        for className in classes:
            prob = []
            for i in range(numAtts):
                try:
                    prob.append(self.posteriors[i][instance[i]][className])
                except:
                    prob.append(1)
            allProbs.append(prob)
        finalProbs = []
        for i, row in enumerate(allProbs):
            total = 1
            for val in row:
                total *= val 
            finalProbs.append(total * self.priors[i]) 
        return finalProbs.index(max(finalProbs))
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        classes = sorted(list(set(self.y_train)))
        predictions = []
        for row in X_test:
            index = self.getSinglePrediction(row)
            predictions.append(classes[index])

        return predictions


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        numAtts = len(X_train[1])
        # Create a header
        header = []
        for i in range(numAtts):
            header.append("att" + str(i))
        
        # determine the attribute domain
        attDomain = {}
        
        for i, att in enumerate(header):
            attDomain[att] = myutils.getUniqueIdentifiers(X_train, i)
        
        # stitch together x and y train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        availableAtts = header.copy()
        
        tree = myutils.tdidt(train, availableAtts, attDomain, header)
        self.tree = tree
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        numAtts = len(X_test[0])
        # Create a header
        header = []
        for i in range(numAtts):
            header.append("att" + str(i))

        predictions = []
        for row in X_test:
            pred = myutils.tdidtPrediction(row, self.tree, header)
            predictions.append(pred)

        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        
        string = 'IF '
        myutils.tdidtPrint(self.tree, string, attribute_names, class_name)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
