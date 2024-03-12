from flask import Flask, request
from flask_restful import Api, Resource
from utils.FileServices import FileServices
import pandas as pd
from core.arachne import *
class CompileModel(Resource):
    def post(self):
        

        return {"message": "Model compiled successfully"}, 200
   
class FitModel(Resource):
    def post(self):
        epochs = request.get_json()['epochs']
        prediction_var = request.get_json()['pred']
        layer_data = request.get_json()['layers']
        
        data = FloatTensor.readCSV("assets/data.csv")
        data = data.Normalize()
        
        df = pd.read_csv("assets/data.csv")
        pred = df.columns.get_loc(prediction_var)
        ind = [pred]
        vals = data.input_output_split(ind)

        input = vals[0]
        output = vals[1]

        # Split the input and output into rows
        input_list = input.row_split()
        output_list = output.row_split()

        myPipeline = Pipeline()
        # input_layer = Linear(IntPair(1, len(input_list)))

        prev_layer = IntPair(1,input_list[0].getSize().second)
        layers = {}
        for layer in layer_data:
            layers[layer['layerId']] = Linear(prev_layer, layer['units'])
            prev_layer = IntPair(prev_layer[0], layer['units'])
            myPipeline.add(layers[layer['layerId']])
            if layer['activation'].lower() == 'relu':
                val = str(layer['layerId']) + 'activaion'
                layers [val] = Relu(prev_layer)
                myPipeline.add(layers[val])
        
        myPipeline.printPipeline()
        optimizer = SGD(1e-4)
        a = MSELoss()
        losses = []
        # Train the model
        for j in range(epochs):
            for i in range(len(input_list)):
                prediction = myPipeline.forwardFloat(input_list[i])
                loss = (a.loss(prediction, output_list[i]))

                myPipeline.backward(optimizer, a, output_list[i])
            print(f"Epoch {j+1}, Loss: {loss}")
            losses.append(loss)
        print(str(losses[-1]))
        msg = f"Model fit successfully, with a loss of {str(losses[-1])}"
        return {"message": msg}, 200