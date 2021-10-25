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

import joblib

def my_main(num):  
    # 1) Select data --> load data
    dataset = pd.read_csv('data_train.csv',sep=',')
    # Printing the dataswet shape
    print("Dataset Length: ",len(dataset))
    print("Dataset Shape: " ,dataset.shape)
    # Printing the dataset obseravtions
    print("Dataset: ",dataset.head())

    # 2) Split data
    X = dataset.values[:,2:5] #row dataset
    Y = dataset.values[:,5] #Class column ที่ 4
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size= 0.3, random_state= 100) #แบ่งข้อมูล 70:30 เรียน:ทดสอบ

    # 3) Model data --> Tree generating
    clf_entropy = DecisionTreeClassifier(criterion= "entropy",
                random_state= 100, max_depth= 3, min_samples_leaf= 5)#สร้างต้นไม้ตัดสินใจ 
    a = clf_entropy.fit(X_train, y_train) #เรียนรู้ข้อมูล
    print("Model : ",a)

    # 4) Function to calculate accuracy
    y_pred = clf_entropy.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    print("\n")
    print("Confusion matrix : ",confusion)
    print("Accuracy : ",accuracy)
    print("Report : ",report)
    # print('จำนวนแถว', num)

    if num == 500:
        filename = 'model_500.ph5'
        joblib.dump(clf_entropy,filename)
    elif num == 400:
        filename = 'model_400.ph5'
        joblib.dump(clf_entropy,filename)

    elif num == 300:
        filename = 'model_300.ph5'
        joblib.dump(clf_entropy,filename)
    elif num == 200:
        filename = 'model_200.ph5'
        joblib.dump(clf_entropy,filename)
    else:
        print("not model!")
        
    return [confusion, accuracy, report] 

