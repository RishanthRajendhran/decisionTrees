import numpy as np
import csv
import sys
import pickle
from validate import validate
import sys

train_X_file_path = "./train_X_de.csv"
train_Y_file_path = "./train_Y_de.csv"
validationSplit = 0.2

def trainValSplit(X,Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    valIndex = -int(validationSplit*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    return (train_X, train_Y, val_X, val_Y)

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

    def setAttrThresh(self, attr, thresh):
        self.feature_index = attr 
        self.threshold = thresh

    def setLeftChild(self, child):
        self.left = child 
        
    def setRightChild(self, child):
        self.right = child
"""
You can make use of the functions
1. get_best_split(X, Y)
    To calculate the best feature 'A' and it's best threshold based on gini index, given data at a node

    Arguments:
    X -- 2D list of floats with shape (num of observations, num of features)
    Y -- 1D list of ints, denoting class values of observations of data_X

    Returns:
    best_feature -- index of the best feature 'A' in X
    best_threshold -- best threshold value of feature 'A'

2. split_data_set(data_X, data_Y, feature_index, threshold)
    To split the data based on the feature 'A', given the threshold 'T'

    Arguments:
    data_X -- 2D list of floats with shape (num of observations, num of features)
    data_Y -- 1D list of ints, denoting class values of observations of data_X
    feature_index -- index of the feature 'A' in data_X to split on
    threshold -- threshold value 'T' to use during split

    Return:
    left_X -- 2D list containing X values of the left subtree (values of A < T), in the same relative order as in data_X
    left_Y -- 1D list containing Y values for left_X
    right_X -- 2D list containing X values of the right subtree (values of A >= T), in the same relative order as in data_X
    right_Y -- 1D list containing Y values for right_X
"""
def calculate_entropy(Y):
    Ys = np.unique(Y)
    entropy = 0 
    N = len(Y)
    for y in Ys:
        num = np.sum(np.array(Y) == y)
        ratio = num/N
        entropy += (-ratio)*np.log2(ratio)
    return entropy

def calculate_information_gain(Y_subsets):
    flatY = []
    for y in Y_subsets:
        flatY.extend(y)
    root = calculate_entropy(flatY)
    numElts = len(flatY)
    for y in Y_subsets:
        root -= (len(y)/numElts)*calculate_entropy(y)
    return root

def calculate_split_entropy(Y_subsets):
    Y = np.concatenate(Y_subsets)
    numElts = len(Y)
    sum = 0
    for y in Y_subsets:
        r = (len(y)/numElts)
        sum -= r*np.log2(r)
    return sum 

def calculate_gain_ratio(Y_subsets):
    return calculate_information_gain(Y_subsets)/calculate_split_entropy(Y_subsets)

def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    classes = sorted(set([j for i in Y_subsets for j in i]))

    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.tolist().count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances)*gini
    
    return gini_index

def split_data_set(data_X, data_Y, feature_index, threshold):
    X = np.array(data_X)
    Y = np.array(data_Y)
    left = X[:,feature_index] < threshold
    right = X[:,feature_index] >= threshold
    return X[left], Y[left], X[right], Y[right]

def get_best_split(X, Y):
    npX = np.array(X)
    best_feature = 0
    best_threshold = 0
    minGini = np.inf
    for feature in range(len(X[0])):
        thresholds = np.unique(npX[:,feature])
        for threshold in thresholds:
            _, leftY, _, rightY = split_data_set(X, Y, feature, threshold)
            gini = calculate_gini_index([leftY, rightY])
            if gini < minGini:
                best_feature = feature 
                best_threshold = threshold
                minGini = gini
                
    return best_feature, best_threshold

def buildTree(X, Y, max_depth, min_size, depth):
    if max_depth <= 0 or min_size == np.inf or depth >= max_depth:
        return None
    curNode = Node(np.max(Y), 0)
    best_feature, best_threshold = get_best_split(X,Y)
    leftX, leftY, rightX, rightY = split_data_set(X,Y,best_feature,best_threshold)
    curNode.setAttrThresh(best_feature, best_threshold)
    if len(leftX) <= min_size:
        curNode.setLeftChild(None)
    else:
        curNode.setLeftChild(buildTree(leftX, leftY, max_depth, min_size, depth+1))
    if len(rightX) <= min_size:
        curNode.setRightChild(None)
    else:
        curNode.setRightChild(buildTree(rightX, rightY, max_depth, min_size, depth+1))
    return curNode

def preorder(node):
    if node == None:
        return
    print(f"X{node.feature_index} {node.threshold}")
    preorder(node.left)
    preorder(node.right)

def predictClass(root, X):
    feature = root.feature_index 
    threshold = root.threshold 
    if X[feature] < threshold:
        if root.left != None:
            return predictClass(root.left, X)
        else: 
            return root.predicted_class
    else:
        if root.right != None:
            return predictClass(root.right, X)
        else: 
            return root.predicted_class

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    Y = []
    for x in test_X:
        Y.append(predictClass(model, x))
    return np.array(Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def trainModel():
    train_X = np.genfromtxt(train_X_file_path, delimiter=",", skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=",", skip_header=0, dtype=np.int64)
    train_X, train_Y, val_X, val_Y = trainValSplit(train_X, train_Y)
    maxDepths= [i for i in range(1,len(train_X[0])+10)]
    minSizes= [i for i in range(len(list(set(train_Y)))+2)]
    maxAcc = 0
    for maxDepth in maxDepths:
        print(f"maxDepth : {maxDepth}")
        for minSize in minSizes:
            root = buildTree(train_X, train_Y, maxDepth, minSize, 0)
            acc = np.sum(np.array(val_Y) == predict_target_values(val_X, root))/len(val_Y)
            print(f"\tminSize : {minSize}, curAcc : {round(acc,3)}, maxAcc : {round(maxAcc,3)}")
            if acc > maxAcc:
                maxAcc = acc 
                bestRoot = root
    return bestRoot

def predict(test_X_file_path):
    if "-trainModel" in sys.argv:
        root = trainModel()
        pickle.dump(root, open("MODEL_FILE.sav", "wb"))
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 