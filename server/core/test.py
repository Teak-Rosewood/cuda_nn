from arachne import FloatTensor, Linear, Pipeline, Relu, MSELoss, SGD, IntPair

a = FloatTensor.randomFloatTensor(IntPair(2,2))
b = FloatTensor(IntPair(2,2),2)
sum = a+b
sum.printTensor()
# Read the CSV file
# dat = FloatTensor.readCSV("WineQT.csv")

# # Normalize the data
# dat = dat.Normalize()

# # # Split the data into input and output
# ind = [11]
# vals = dat.input_output_split(ind)
# input = vals[0]
# output = vals[1]

# # Split the input and output into rows
# input_list = input.row_split()
# output_list = output.row_split()

# # Create the pipeline
# myPipeline = Pipeline()
# size = IntPair(1,12)
# q = Linear(size,6)
# r = Relu(IntPair(1,6))
# d = Linear(IntPair(1,6),3)
# e = Relu(IntPair(1,3))
# f = Linear(IntPair(1,3),1)
# g = Relu(IntPair(1,1))

# # Add the layers to the pipeline
# myPipeline.add(q)
# myPipeline.add(r)
# myPipeline.add(d)
# myPipeline.add(e)
# myPipeline.add(f)
# myPipeline.add(g)

# myPipeline.load("help.arachne")
# # Print the pipeline
# myPipeline.printPipeline()

# # Create the optimizer
# optimizer = SGD(1e-4)

# a = MSELoss()

# # Train the model
# for j in range(10):
#     for i in range(len(input_list)):
#         prediction = myPipeline.forwardFloat(input_list[i])

#         loss = a.loss(prediction, output_list[i])

#         # myPipeline.backward(optimizer, a, output_list[i])
#     print(f"Epoch {j+1}, Loss: {loss}")
# # myPipeline.save("help")
