import json
import os

from flask import Flask, jsonify, request
from connecttion import get_database_by_name
from pprint import pprint
import csv
from decisiontree import my_main

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from sklearn import tree
from matplotlib import pyplot as plt
import graphviz
import joblib

app = Flask(__name__)

def create_csv_train(num):
    db = get_database_by_name('Roomsystem')
    col_static = db['static']
    # sum = num
    # print("จำนวน record ==> ",sum)
    statics = col_static.find().limit(num)
    header = ['_id', 'datetime', 'luminance', 'motion', 'temperature', 'status']
    with open('data_train.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for static in statics:
            # print(static['_id'])
            writer.writerow(
                [static['_id'],
                 static['datetime'],
                 static['luminance'],
                 static['motion'],
                 static['temperature'],
                 'Full' if static['label'] == 'กำลังถูกใช้' else 'Empty'
                 ]
            )

@app.route('/trainnig/<int:num>')
def train_ning(num):
    # sum = num
    # print("จำนวน record ==> ",sum)
    [confusion, accuracy, report] = my_main(num)
    create_csv_train(num)
    json_str = json.dumps(
        {'message': 'OK','Record data':num ,'confusion': confusion, 'accuracy': accuracy, 'report': report},
        ensure_ascii=False, default=str)
    return app.response_class(json_str, mimetype='application/json')

@app.route('/prediction/<int:number>')
def hello_world(number):
    db = get_database_by_name('Roomsystem')
    col_infomation1 = db['information']
    informotions1 = col_infomation1.find()
    # print("==>",informotions1)
    count_data = informotions1.count()
    # count_data = col_infomation1.count_documents({})
    print(count_data)
    datasensor = np.zeros((count_data,3))
    # print(datasensor)
    data = []

    # #เรียกโมเดล
    if number == 5:
        path_model = 'model_500.ph5'
    elif number == 4:
        path_model = 'model_400.ph5'
    elif number == 3:
        path_model = 'model_300.ph5'
    elif number == 2:
        path_model = 'model_200.ph5'
    # path_model = 'model_500.ph5'
    loaded_model = joblib.load(path_model)

    for i,data in enumerate(informotions1):
        datasensor[i][0] = data['luminance']
        datasensor[i][1] = data['motion']
        datasensor[i][2] = data['temperature']

    currentId = data['_id']
    currentId1 = data['luminance']
    currentId2 = data['motion']
    currentId3 = data['temperature']
    print(currentId)
    print(currentId1)
    print(currentId2)
    print(currentId3)
    #last index
    value = datasensor[count_data-1]
    # print(value)
    
    re = ""
    status = 0
    va = value.reshape(1,-1)
    pred = loaded_model.predict(va)
    listToStr = ' '.join([str(elem) for elem in pred])
    print(listToStr)

    #update MongoDB
    col_infomation1.update_many({'_id': currentId},{"$set":{"label": listToStr}}) 
    re = listToStr
    if re == "Empty":
        status = 0
    else:
        status = 1

    json_str = json.dumps(
        {'message': 'OK','Model':number,'Result':listToStr, 'status': status})
    return app.response_class(json_str, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=20000)
