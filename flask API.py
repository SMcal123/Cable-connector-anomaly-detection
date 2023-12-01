from flask import Flask, render_template, request

import logging as logger
logger.basicConfig(level="DEBUG")

from .api.cable_anomaly_detection import CablePhotoCollect

import sqlite3

flaskAppInstance = Flask(__name__)

@flaskAppInstance.route('/', methods = ['GET'])
def index():
    logger.debug("Starting Flask Server")
    return render_template('index.html')


@flaskAppInstance.route('/api', methods = ['POST'])
def cable_api_call():
    form_data = request.files
    print(form_data)
    logger.debug('Calling the API.')
    result = CablePhotoCollect().anomaly_detection(form_data)
    print(result)
    return result

if __name__ == '__main__':
    flaskAppInstance.run(host="127.0.0.1",port=5000,debug=True,use_reloader=True)
