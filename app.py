import json
import os

from flask import Flask, jsonify, request
from connecttion import get_database_by_name
from pprint import pprint
# import csv
# from decisiontree import my_main

# import pandas as pd
import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.tree import export_text

# from sklearn import tree
# from matplotlib import pyplot as plt
# import graphviz
import joblib

app = Flask(__name__)

@app.route('/prediction')
def hello_world():  # put application's code here
    db = get_database_by_name('Roomsystem')
    col_infomation1 = db['information']
    informotions1 = col_infomation1.find()
    # print("==>",informotions1.count())
    count_data = informotions1.count()
    # print(count_data)
    datasensor = np.zeros((count_data,3))
    # print(datasensor)
    data = []

    # #เรียกโมเดล
    path_model = 'jubjang_model.pp'
    loaded_model = joblib.load(path_model)

    for i,data in enumerate(informotions1):
        datasensor[i][0] = data['luminance']
        datasensor[i][1] = data['motion']
        datasensor[i][2] = data['temperature']
    currentId = data['_id']
    #last index
    value = datasensor[count_data-1]
    
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
        {'message': 'OK','Result':listToStr, 'status': status})
    return app.response_class(json_str, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=20000)
