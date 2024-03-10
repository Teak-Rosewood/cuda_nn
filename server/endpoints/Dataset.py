from flask import Flask, request
from flask_restful import Api, Resource
from utils.FileServices import FileServices
import pandas as pd

class UploadDataset(Resource):
    def post(self):
        save_directory = 'assets'
        file = request.files['file']
        result = FileServices.saveFile(file, save_directory)
        if(result):
            df = pd.read_csv("assets/data.csv")
            attributes = df.columns.tolist()
            return {"features": attributes}, 200
        else: 
            return {"message": "Error saving file"}, 500

class DatasetFeatures(Resource):
    def get(self):
        try:
            df = pd.read_csv("assets/data.csv")
            attributes = df.columns.tolist()
            return {"features": attributes}, 200
        except:
            return {"message": "Error reading file"}, 500