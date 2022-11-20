import math
import numpy as np 
import matplotlib.pyplot as plt
import copy
import operator 
import random
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 


def compute_slope_intercept(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    slope = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b

    b = mean_y - mean_x * slope

    return slope, b



def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA" or "N/A")
        if row[col_index] != 'NA' and row[col_index] != 'N/A':
            col.append(row[col_index])
    return col

def get_min_max(values):

    return min(values), max(values)

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_frequencies_categorical(table, header, col_name):
    col = get_column(table, header, col_name)

    for i in range(len(col)):
        col[i] = str(col[i])

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)

    # get a list of unique values for the column
    group_names = sorted(list(set(col))) # 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], []]

    # walk through each row and assign it to the appropriate
    # subtable based on its group by value (model year)
    for row in table:
        group_value = row[col_index]
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables


def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]
    print(freqs)
    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1
                #elif cutoffs[i] < val < cutoffs[i + 1]:
                    #freqs[i] += 1

    return freqs


def determine_frequencies(values, bins):

    frequencies = [0] * len(bins) # initialize frequencies to zero
    # check which bin the value falls into
    for value in values:
        for i in range(len(bins) - 1):
            if value >= bins[-1][0]: # found value that falls in last bin
                frequencies[-1] += 1
                break
            else: # check for all other cases
                if value >= bins[i][0] and value < bins[i][1]:
                    frequencies[i] += 1
                    break
                elif value >= bins[i][1] and value < bins[i + 1][0]: # value falls between bins
                    frequencies[i] += 1
                    break 
    return frequencies

def convert_bins_to_str(bins):
    strBins = []
    for aBin in bins:
        strBins.append(str(aBin))

    return strBins

def compute_equal_width_cutoffs(values, num_bins):
    values_range = max(values) - min(values) # range of the values
    bin_width = values_range / num_bins # equal width bins 

    cutoffs = list(np.arange(min(values), max(values), bin_width)) # determine the cutoffs 
    cutoffs.append(max(values)) # add the max as the last cutoff
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs] # round each cutoff to two decimal places
    return cutoffs 

def convert_cutoffs_to_bins(cutoffs):
    bins = []
    
    for i in range(len(cutoffs) - 1):
        aBin = []
        aBin.append(cutoffs[i])
        aBin.append(cutoffs[i + 1])
        bins.append(aBin)
    return bins

def compute_covariance(x,y):
    xBar = np.mean(x) 
    yBar = np.mean(y) 
    n = len(x) 

    cov = sum((x - xBar) * (y - yBar)) / n
    return cov 

def compute_correlation_coefficient(x,y):
    xBar = np.mean(x) 
    yBar = np.mean(y) 

    r = sum((x - xBar) * (y - yBar)) / math.sqrt(sum((x - xBar) ** 2) * sum((y - yBar) ** 2))
    return r

def remove_unwanted_character(data, character):
    for i in range(len(data)):
        newVal = float(data[i].strip(character))
        data[i] = newVal 
    
def genre_rating(dataset, header, genre, website):
    gIndex = header.index('Genres')
    webIndex = header.index(website)

    ratings = []

    for row in dataset:
        if genre in row[gIndex] and row[webIndex] != '':
            ratings.append(row[webIndex]) 
    return ratings


def convert_2D_to_1D(data): # converts 2D list to 1D list
    newList = []
    if type(data[0]) == list:
        for i in range(len(data)):
            for j in range(len(data[i])):
                newList.append(data[i][j])
        return newList 

def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)
    if isinstance(v1[0], str):
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                return 1
        return 0

    dist = np.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1)))) # euclidean distance formula
    return dist 


def getKNeighbors(test, trainData, k):
    xTrain = copy.deepcopy(trainData)
    for i, row in enumerate(xTrain):
        dist = compute_euclidean_distance(test, row)
        row.append(i)
        row.append(round(dist, 3))

    sortedList = sorted(xTrain, key=operator.itemgetter(-1))
    allDist = []
    allIndex = []
        
    for j in range(0, k):
        allDist.append(sortedList[j][-1])
        allIndex.append(sortedList[j][-2])
    return allDist, allIndex 


def getClassification(classifications):
    values = []
    counts = []

    classifications.sort()

    for value in classifications:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    maxIndex = counts.index(max(counts))

    return values[maxIndex]

def randomize_in_place(aList, parallel_list=None):
    for i in range(len(aList)):
        randIndex = random.randrange(0,len(aList)) 
        aList[i], aList[randIndex] = aList[randIndex], aList[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[randIndex] = parallel_list[randIndex], parallel_list[i] 

def kFoldGroupBy(data):
    # get a list of unique values
    vals = []
    for row in data:
        vals.append(row[-1])

    
    group_names = sorted(list(set(vals)))
    
    group_subtables = [[] for _ in group_names] 

    # walk through each row and assign it to the appropriate
    # subtable based on its group by value 
    for row in data:
        group_value = row.pop(-1)
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables


def determineBin(value, bins):
    for i in range(len(bins) - 1):
        if value >= bins[-1][0]: # found value that falls in last bin
            return len(bins) - 1
        else: # check for all other cases
            if value >= bins[i][0] and value < bins[i][1]:
                return i
            elif value >= bins[i][1] and value < bins[i + 1][0]: # value falls between bins
                return i

def normalizeTable(data):
    newTable = []
    for row in data:
        newTable.append([])

    for i in range(len(data[0])):
        newCol = []
        for row in data:
            newCol.append(row[i])
        minVal = min(newCol)
        for j in range(len(newCol)):
            newCol[j] -= minVal
        maxVal = max(newCol)
        for k in range(len(newCol)):
            newCol[k] = newCol[k] / maxVal
        for m in range(len(newCol)):
            newTable[m].append(round(newCol[m], 2))
    return newTable

def createConfusionMatrix(yTrue, yTest, header, categories):
    table = MyPyTable()
    table.column_names = header 
    table.data = []

    for val in categories:
        newRow = [val]
        for i in range(len(header) - 1):
            newRow.append(0)
        table.data.append(newRow)

    for i in range(len(yTrue)):
        table.data[yTrue[i] - 1][yTest[i]] += 1
    for row in table.data:
        total = 0
        for i in range(1, len(categories) + 1):
            total += row[i]
        row[len(categories) + 1] = total

    for i in range(len(table.data)):
        if table.data[i][len(categories) + 1] != 0:
            recognition = table.data[i][i + 1] / table.data[i][len(categories) + 1]
            table.data[i][len(header) - 1] = round(100 * recognition, 2)
    return table

def confusionCategorical(yTrue, yTest, header, categories):
    table = MyPyTable()
    table.column_names = header 
    table.data = []

    for val in categories:
        newRow = [val]
        for i in range(len(header) - 1):
            newRow.append(0)
        table.data.append(newRow)

    for i in range(len(yTrue)):
        rowIndex = categories.index(yTrue[i])
        colIndex = header.index(yTest[i])
        table.data[rowIndex][colIndex] += 1

    for row in table.data:
        total = 0
        for i in range(1, len(categories) + 1):
            total += row[i]
        row[len(categories) + 1] = total

    for i in range(len(table.data)):
        if table.data[i][len(categories) + 1] != 0:
            recognition = table.data[i][i + 1] / table.data[i][len(categories) + 1]
            table.data[i][len(header) - 1] = round(100 * recognition, 2)
    return table


def groupBy(data):
    # get a list of unique values
    vals = []
    for row in data:
        if type(row) != list:
            vals.append(row)
        else:
            vals.append(row[0])
    
    group_names = sorted(list(set(vals)))
    
    group_subtables = [[] for _ in group_names] 
    # walk through each row and assign it to the appropriate
    # subtable based on its group by value 
    for row in data:
        if type(row) == list:
            group_value = row.pop(0)
            # which group_subtable??
            group_index = group_names.index(group_value)
            group_subtables[group_index].append(row[0])
    return group_names, group_subtables


def getPriors(classes):
    group_names = sorted(list(set(classes)))
    priors = []
    
    for val in group_names:
        count = 0
        for vals in classes:
            if val == vals:
                count += 1
        priors.append(round(count / len(classes), 3))
    return group_names, priors

def getUniqueIdentifiers(data, index):
    column = []
    for row in data:
        column.append(row[index])
    return sorted(list(set(column)))

def mostFrequentClass(col):

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values[counts.index(max(counts))]


def convertWeightToCategorical(weight):
    if weight <= 1999:
        return 1
    elif weight > 2000 and weight < 2499:
        return 2
    elif weight > 2500 and weight < 2999:
        return 3
    elif weight > 3000 and weight < 3499:
        return 4
    else:
        return 5

def allSameClass(data):
    firstLabel = data[0][-1]
    for instance in data:
        if instance[-1] != firstLabel:
            return False 
    return True # if we get here, all instance labels matched the first label

def selectAtt(data, availableAtts):
    indices = []
    for att in availableAtts:
        indices.append(int(att[-1]))

    entropies = []
    for i in indices:
        entropies.append(computeEntropyOneAtt(data, i))
    #print('available', availableAtts)
    #print('entropies', entropies)
    return availableAtts[entropies.index(min(entropies))]
    
    
    

def tdidt(currInstances, availableAtts, attDomain, header):
    splitAtt = selectAtt(currInstances, availableAtts)
    availableAtts.remove(splitAtt)

    tree = ['Attribute', splitAtt]

    prevPartition = currInstances
    partitions = partitionInstances(currInstances, splitAtt, attDomain, header)

    for attVal, partition in partitions.items():
        valSubtree = ['Value', attVal]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and allSameClass(partition):
            #print("CASE 1")
            majorityRule, num, total = determineMajority(partition)
            valSubtree.append(['Leaf', partition[0][-1], num, len(currInstances)])
            tree.append(valSubtree)


        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(availableAtts) == 0:
            #print("CASE 2")
            majorityRule, num, total = determineMajority(partition)
            valSubtree.append(['Leaf', majorityRule, num, total])
            tree.append(valSubtree)


        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            majorityRule, num, total = determineMajority(prevPartition)
            tree = ['Leaf', majorityRule, num, total]

        else: # all base cases are false, recurse!!
            #print('CASE 4')
            valTree = ['Value', attVal]
            subtree = tdidt(partition, availableAtts.copy(), attDomain, header)
            valTree.append(subtree)
            tree.append(valTree)
    return tree


def partitionInstances(instances, splitAtt, attDomain, header):
    attDomain = attDomain[splitAtt]
    attIndex = header.index(splitAtt)
    
    partitions = {}
    for attValue in attDomain:
        partitions[attValue] = []
        for instance in instances:
            if instance[attIndex] == attValue:
                partitions[attValue].append(instance)
    return partitions

def determineMajority(instances):
    classes = getUniqueIdentifiers(instances, -1)
    numEachClass = [0 for i in range(len(classes))]

    for instance in instances:
        for i in range(len(classes)):
            if instance[-1] == classes[i]:
                numEachClass[i] += 1
    return classes[numEachClass.index(max(numEachClass))], max(numEachClass), len(instances)

def computeEntropyOneAtt(data, index):
    # create a 2D array of values of attribute and their corresponding classes
    newData = []
    for row in data:
        instance = []
        instance.append(row[index])
        instance.append(row[-1])
        newData.append(instance)
    
    attNames, attClasses = groupBy(copy.deepcopy(newData)) # get unique att values and classes assoc. w those values

    # get number of instances w each att value
    attCounts = [0 for att in attNames]
    for i, att in enumerate(attNames):
        for row in newData:
            if att == row[0]:
                attCounts[i] += 1
    # weight the counts
    for i in range(len(attCounts)):
        attCounts[i] /= len(newData)

    priors = [] 
    # get priors for each att value
    for i, att in enumerate(attNames):
        #print("CLASSES", attClasses[i])
        names, pr = getPriors(attClasses[i])
        priors.append(pr)
    entropies = [0 for att in attNames] # initialize entropies to zero

    # go through each att value and calculate entropy
    for i, vals in enumerate(priors):
        for pr in vals:
            entropies[i] -= pr * math.log(pr, 2)

    # weight the entropies
    for i in range(len(entropies)):
        entropies[i] *= attCounts[i] 
    weightedEntropy = sum(entropies) # get weighted sum

    return weightedEntropy

def tdidtPrediction(instance, tree, header):
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidtPrediction(instance, value_list[2], header)
    else: # Leaf
        return tree[1] # label

def tdidtPrint(tree, string, atts=None, className = 'class'):

    if atts is not None:
        index = int(tree[1][-1])
        string += str(atts[index]) + ' == '
    else:
        string += str(tree[1]) + ' == ' 
    strCopy = copy.deepcopy(string) # needed for each subtree
    
    for i in range(2, len(tree)): # visit all subtrees
        subtree = tree[i] # get subtree
        string += str(subtree[1])
        
        # base case. found a leaf, append value and print
        if subtree[2][0] == 'Leaf':
            string +=' THEN ' + str(className) + ' = ' + str(subtree[2][1])
            print(string)
     
        # not at a leaf, need to recurse
        else:
            string += ' AND '
            tdidtPrint(subtree[2], string, atts, className)
       
        # visiting next subtree, dont want the other subtrees classification rules
        string = strCopy 
        