import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from sklearn import tree
# from matplotlib import pyplot as plt
import graphviz
import os


def import_data():
    dataset = pd.read_csv('data//data_train.csv', sep=',')

    # Printing the dataswet shape
    print("Dataset Length: ", len(dataset))
    print("Dataset Shape: ", dataset.shape)
    # Printing the dataset obseravtions
    print("Dataset: ", dataset.head())
    return dataset


# Function to split the dataset
def split_dataset(dataset):
    # Separating the target variable
    X = dataset.values[:, 2:5]  # row dataset
    Y = dataset.values[:, 5]  # Class column ที่ 4

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


def train_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy",
                                         random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion vatrix : ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


def my_main():
    # Building Phase
    data = import_data()
    # print("==> ",data)
    X, Y, X_train, X_test, y_train, y_test = split_dataset(data)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("<------------------------------ Result Using Entropy ---------------------------->")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

#region prediction old
    # Target_names = ['Full', 'Empty']
    # Feature_names = ['luminance', 'motion', 'temperature']
    # dot_data = tree.export_graphviz(clf_entropy, out_file=None, feature_names=Feature_names,
    #                                 class_names=Target_names, filled=True, rounded=True,
    #                                 special_characters=True)  # Filled คือ เติมสี, Rounded คือ ทำขอบมน
    # graph = graphviz.Source(dot_data)
    # graph.render('/content/drive/MyDrive/Python/Tree/dtree_render1', view=True)
    # tree.export_graphviz(clf_entropy, out_file='/content/drive/MyDrive/Python/Tree/Tree.dot1')
    # graph.render('dtree_render1', view=True)
    # tree.export_graphviz(clf_entropy, out_file='Tree.dot1')
    #
#endregion
    
    # # 4) Use the model
    confusion = confusion_matrix(y_test, y_pred_entropy)
    accuracy = accuracy_score(y_test, y_pred_entropy) * 100
    report = classification_report(y_test, y_pred_entropy)

    dataTest = pd.read_csv('data//data_test.csv', sep=',')
    modelUse = dataTest.values[:, 2:5]  # row dataset
    Result = clf_entropy.predict(modelUse)
    print(Result)
    return [Result, confusion, accuracy, report]
