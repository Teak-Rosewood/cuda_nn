from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_restful import Api, Resource

from endpoints.Dataset import UploadDataset
from endpoints.Model import CompileModel, FitModel

import os 
import re

class SetupEndpoints:
    api = None
    def __init__(self, app):
        self.api = Api(app)
        app.config['SECRET_KEY'] = 'mysecret'
        app.host = '0.0.0.0'
        app.port = 5000
        app.debug = True
        self.addResources()

    def addResources(self):
        self.api.add_resource(UploadDataset, '/api/upload')
        self.api.add_resource(CompileModel, '/api/compile')
        self.api.add_resource(FitModel, '/api/fit')

app = Flask(__name__)
SetupEndpoints(app)
CORS(app)

if __name__ == '__main__':
    socketio = SocketIO(cors_allowed_origins='*')
    socketio.init_app(app)
    try:
        socketio.run(app)
    except:
        socketio.run(app)