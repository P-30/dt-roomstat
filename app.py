import json
import os

from flask import Flask, jsonify, request
from connecttion import get_database_by_name
from pprint import pprint
import csv
from decisiontree import my_main

app = Flask(__name__)

header = ['_id', 'datetime', 'luminance', 'motion', 'temperature', 'status']


def create_csv_train():
    db = get_database_by_name('Roomsystem')
    col_static = db['static']
    statics = col_static.find().limit(500)
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

def create_csv_test():
    db = get_database_by_name('Roomsystem')
    col_infomation = db['information']
    informotions = col_infomation.find()
    header = ['_id', 'datetime', 'luminance', 'motion', 'temperature']
    with open('data_test.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        writer.writerow(header)
        for information in informotions:
            writer.writerow(
                [information['_id'],
                 information['datetime'],
                 information['luminance'],
                 information['motion'],
                 information['temperature']
                ]
            )


@app.route('/prediction')
def hello_world():  # put application's code here
    db = get_database_by_name('Roomsystem')
    col_infomation1 = db['information']
    informotions1 = col_infomation1.find()

    create_csv_test()
    create_csv_train()
    [result, confusion, accuracy, report] = my_main()
    result = result.tolist()
    print(result)

    for x,data in zip(informotions1,result):
        col_infomation1.update_many({'_id': x['_id']},{"$set":{"label": data}})

    json_str = json.dumps(
        {'message': 'OK', 'result': result, 'confusion': confusion, 'accuracy': accuracy, 'report': report,
            'status': 1},
        ensure_ascii=False, default=str)
    return app.response_class(json_str, mimetype='application/json')
    # return jsonify({'msg': 'Hello World!'})


#region prediction old
# @app.route('/prediction', methods=['POST'])
# def train():
#     if request.method == "POST":
#         if 'file' in request.files:
#             file = request.files['file']
#             file.save(os.path.join('', 'file.csv'))

#             create_csv()
#             [result, confusion, accuracy, report] = my_main('file.csv')
#             result = result.tolist()
#             json_str = json.dumps(
#                 {'message': 'OK', 'result': result, 'confusion': confusion, 'accuracy': accuracy, 'report': report,
#                  'status': 1},
#                 ensure_ascii=False, default=str)
#             return app.response_class(json_str, mimetype='application/json')
#         else:
#             json_str = json.dumps({'result': [], 'message': 'required file', 'status': 0},
#                                   ensure_ascii=False, default=str)
#             return app.response_class(json_str, mimetype='application/json')
#     else:
#         json_str = json.dumps({'message': '405 Method Not Allowed', 'status': 0},
#                               ensure_ascii=False, default=str)
#         return app.response_class(json_str, mimetype='application/json')
#endregion

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=20000)
