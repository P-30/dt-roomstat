import json
import os

from flask import Flask, jsonify, request
from connecttion import get_database_by_name
from pprint import pprint
import csv
from decisiontree import my_main

app = Flask(__name__)

header = ['_id', 'datetime', 'luminance', 'motion', 'temperature', 'status']


def create_csv():
    db = get_database_by_name('Roomsystem')
    col_static = db['static']
    statics = col_static.find().limit(500)
    with open('static.csv', 'w', encoding='UTF8') as f:
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


@app.route('/')
def hello_world():  # put application's code here

    return jsonify({'msg': 'Hello World!'})


@app.route('/prediction', methods=['POST'])
def train():
    if request.method == "POST":
        file = request.files['file']
        file.save(os.path.join('', 'file.csv'))

        create_csv()
        [result, confusion, accuracy, report] = my_main('file.csv')
        result = result.tolist()
        json_str = json.dumps({'result': result, 'confusion': confusion, 'accuracy': accuracy, 'report': report},
                              ensure_ascii=False, default=str)
        return app.response_class(json_str, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=20000)
