from flask import Flask, request
from flask_restful import Api, Resource
from utils.FileServices import FileServices
import pandas as pd
import tensorflow as tf


class CompileModel(Resource):
    def post(self):
        layer_data = request.get_json()['layers']
        model = tf.keras.Sequential()
        df = pd.read_csv("assets/data.csv")
        attributes = df.columns.tolist()
        if(layer_data[0]['units'] != len(attributes)-1):
            return {"message": "Input layer units must match number of features"}, 400
        if(layer_data[-1]['units'] != 1):
            return {"message": "Output layer units must be 1"}, 400
        for layer in layer_data:
            model.add(tf.keras.layers.Dense(layer['units'], activation=layer['activation'].lower()))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        model.build((None, len(attributes)-1))
        model.save('assets/model.keras')
        return {"message": "Model compiled successfully"}, 200
   
class FitModel(Resource):
    def post(self):
        epochs = request.get_json()['epochs']
        prediction_var = request.get_json()['pred']
        
        df = pd.read_csv("assets/data.csv")
        X = df.drop(prediction_var, axis=1)
        y = df[prediction_var]
        print(X.shape, y.shape)
        model = tf.keras.models.load_model('assets/model.keras')
        history = model.fit(X, y, epochs=epochs)
        model.save('assets/model.keras')
        final_loss = history.history['loss'][-1]
        return {"message": "Model fit successfully", "final_loss": final_loss}, 200