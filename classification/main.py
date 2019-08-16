from __future__ import division
from __future__ import print_function
from StringIO import StringIO
import sys
import os
import csv
import math
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def read_features_from_csv(features_path):
    features = []
    csvfile = open(features_path, 'rb')
    reader = csv.reader(csvfile)
    row = next(reader) # Drop column names
    for row in reader:
        row = str(row[0])
        row = row.split('; ')
        feature_size = int(row[0])
        feature = [0] * feature_size
        for i in range(feature_size):
            feature[i] = float(row[i + 1])
        features.append(feature)    
    return features


def read_data_from_csv(features_path):
    features = []
    csvfile = open(features_path, 'rb')
    reader = csv.reader(csvfile)
    target = []
    #row = next(reader) # Drop column names
    for row in reader:
        row = str(row[0])
        row = row.split('; ')
        feature_size = 7
        feature = [0] * feature_size
        for i in range(feature_size):
            a = float(row[i])
        features.append(feature)
        target.append(int(row[7].replace(';','')))

    return [features, target]



def train(training_path_a, training_path_b, max_iter=10):
    print ("Reading features")
    training_a = read_features_from_csv(training_path_a)
    training_b = read_features_from_csv(training_path_b)
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [1] * len(training_a) + [0] * len(training_b)
    #[data, target] = read_data_from_csv(training_path_a)
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, probability = True),
        SVC(gamma=2, C=1, probability = True),
        SVC(kernel="rbf", C=1000, gamma=0.01),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=100, max_features=1),
        AdaBoostClassifier(),
        GaussianNB()]
    
    classifiersScore = {
        "Nearest Neighbors" : 0.0, 
        "Linear SVM" : 0.0, 
        "RBF SVM" : 0.0, 
        "SVM day&night" : 0.0,
        "Decision Tree" : 0.0, 
        "Random Forest" : 0.0, 
        "AdaBoost" : 0.0,
        "Naive Bayes" : 0.0
    }

    print ("Training classifiers")
    for iter in range(max_iter):
        print ("Start iteration %d/%d" % (iter, max_iter))
        # split training data in a train set and a test set. The test set will
        # containt 20% of the total
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data,target, test_size=0.20)
        
        for name, classifier in zip(classifiersScore.keys(), classifiers):
            classifier.fit(x_train, y_train)
            score = classifier.score(x_test, y_test)
            classifiersScore[name] += score

    print ("Classifiers score")
    for name in classifiersScore:
        print ('%s = %.6f' % (name, classifiersScore[name] / max_iter))

if len(sys.argv) == 3:
    train(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 4:
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]))
else:
    print ("Wrong arguments!\nUsage: python classifier.py features_a.csv features_b.csv [max_iter]")
