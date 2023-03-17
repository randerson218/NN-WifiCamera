import numpy as np
from activations import Sigmoid,Tanh
import math
from layer import Layer
from losses import mse, mse_prime
import random
import matplotlib.pyplot as plt
import os
import time

#PARAMETERS
lst1 = os.listdir("./inputdata/person") 
lst2 = os.listdir("./inputdata/notperson") 

NUM_DATA_POINTS = int(len(lst1) + len(lst2))
EPOCHS = 1000
LEARNING_RATE = 0.001

#Holds truth vals
trainingTruth = []

#read csvs into traininginputs as an array of arrays (100 element array for each input)
trainingInputs = []


for filename in os.listdir("./inputdata/person"):
     if ".csv" in filename:
        with open("./inputdata/person/" + filename) as csvfile:
            trainingInputs.append(csvfile.read().strip().replace("\n",",").split(","))
            trainingTruth.append(1)
for filename in os.listdir("./inputdata/notperson"):
     if ".csv" in filename:
        with open("./inputdata/notperson/" + filename) as csvfile:
            trainingInputs.append(csvfile.read().strip().replace("\n",",").split(","))
            trainingTruth.append(0)

#reshape data to appropriate dimensions
trainInputReshaped = np.reshape(trainingInputs, (NUM_DATA_POINTS, 100, 1))
trainTruthReshaped = np.reshape(trainingTruth, (NUM_DATA_POINTS, 1, 1))
trainInputReshaped = np.array(trainInputReshaped, dtype=int)
trainTruthReshaped = np.array(trainTruthReshaped, dtype=int)


#BUILD NETWORK
network = [
    Layer(100,20),
    Sigmoid(),
    Layer(20,1) 
]

errfile = open("error.csv","w")

#TRAIN
for e in range(EPOCHS):
    error = 0
    for x, y in zip(trainInputReshaped,trainTruthReshaped):
        #forward prop
        output = x
        for layer in network:
            output = layer.forward(output)

        #error
        error += mse(y,output)
            

        #backward prop
        grad = mse_prime(y,output)
        for layer in reversed(network):
            grad = layer.backward(grad, LEARNING_RATE)
        
    error /= len(trainInputReshaped)
    errfile.write("%s,%s\n"%(e,error))
    print("%d/%d, error=%f"%(e+1,EPOCHS,error))

errfile.close()
print("TRAINING FINISHED")

##print weights of connections after training
#print("IH Weights:\n",network[0].weights,"\n")
#print("HO Weights:\n",network[2].weights, "\n")


#TESTING SECTION
start = time.perf_counter()

testCases = []
testTruth = []

for filename in os.listdir("./tests/person"):
     if ".csv" in filename:
        with open("./tests/person/" + filename) as csvfile:
            testCases.append(csvfile.read().strip().replace("\n",",").split(","))
            testTruth.append(1)

for filename in os.listdir("./tests/notperson"):
     if ".csv" in filename:
        with open("./tests/notperson/" + filename) as csvfile:
            testCases.append(csvfile.read().strip().replace("\n",",").split(","))
            testTruth.append(0)

numCases = len(testCases)

testCasesReshaped = np.reshape(testCases, (numCases, 100, 1))
testTruthReshaped = np.reshape(testTruth, (numCases, 1, 1))
testCasesReshaped = np.array(testCasesReshaped, dtype=int)
testTruthReshaped = np.array(testTruthReshaped, dtype=int)

inputCounter = 0
testoutputs = []
for x, y in zip(testCasesReshaped,testTruthReshaped):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        testoutputs.append(output[0][0])
        print("Actual: %s, Predicted: %s\n" % (testTruth[inputCounter],output[0][0]))
        inputCounter +=1

#Time per prediction of test cases
#print("Average Time Per Prediction: %f" % ((time.perf_counter()-start)/numCases))