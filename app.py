from flask import Flask, jsonify, request
import numpy as np
import pickle
from keras.models import model_from_json
import pandas as pd
import datetime
import re
from final_file import final

import flask
app = Flask(__name__)

final_object=final()

@app.route('/',methods=['GET'])
def home():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    if features[0].isdigit() == True:
         
        
        if features[1] not in pd.date_range(start='2015-07-06', end='2017-09-10'):
            return flask.render_template('index.html',error_date="Enter a Correct Date between 2015-07-06 to 2017-09-10")
        else:
            if not 0 <= int(features[0]) <= 9999:
                return flask.render_template('index.html',error_index="Enter a Correct Index value between 0-9999")
            else:
                client,access,language,predicted,time,page=final_object.predict(features[0],features[1])
                return flask.render_template('new.html',Client=client,Access=access,Language=language,predicted=predicted,time=time,Page=page)
    else:
        return flask.render_template('index.html',error_index="Enter a Correct Integer Index value")
        
if __name__ == '__main__':
    app.run(debug=True)
