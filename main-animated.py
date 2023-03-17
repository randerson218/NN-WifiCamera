import numpy as np
from activations import Sigmoid
import math
from layer import Layer
from losses import mse, mse_prime
import random
import matplotlib.pyplot as plt

#PARAMETERS

#number of data points per set
NUM_DATA_POINTS = 500
EPOCHS = 500
LEARNING_RATE = 0.0001

#GENERATING DATA SECTION

#Holds X vals
trainingX = []
#Holds Y vals
trainingY = []
#Holds truth vals
trainingTruth = []

#Contains all possible numbers
dataset = []

#create datapool from -5 to 5 with NUM_DATA_POINTS points
dataVal = -5
while dataVal <= 5:
    dataset.append(dataVal)
    dataVal += 10/((NUM_DATA_POINTS*2)-1)

random.shuffle(dataset)

#GENERATE TRAINING DATA SECTION:
#generate exclusive random points for training
#y uses same data values but shuffled (so its not just x*x)
for i in range(NUM_DATA_POINTS):
    val = dataset.pop()
    trainingX.append(val)

trainingY = trainingX.copy()
random.shuffle(trainingY)

#generate truth values
for i in range(len(trainingX)):
    trainingTruth.append(trainingX[i]*trainingY[i])

#generate [X,Y] pairs
trainingInputs = np.column_stack((trainingX,trainingY))

#reshape data to appropriate dimensions
trainInputReshaped = np.reshape(trainingInputs, (NUM_DATA_POINTS, 2, 1))
trainTruthReshaped = np.reshape(trainingTruth, (NUM_DATA_POINTS, 1, 1))

#GENERATE TESTING DATA SECTION:

#grab the rest of the values in the dataset for X
testingX = dataset.copy()
testingY = testingX.copy()

random.shuffle(testingY)

testingTruth = []
for i in range(len(trainingX)):
    testingTruth.append(trainingX[i]*trainingY[i])

testingInputs = np.column_stack((testingX,testingY))



#BUILD NETWORK
network = [
    Layer(2,10),
    Sigmoid(),
    Layer(10,1)
]

errfile = open("error.csv","w")

plt.ion()
trainFig = plt.figure()
ax = trainFig.add_subplot(111, projection="3d")


#TRAIN
for e in range(EPOCHS):
    error = 0
    trainoutputs = []
    for x, y in zip(trainInputReshaped,trainTruthReshaped):
        #forward prop
        output = x
        for layer in network:
            output = layer.forward(output)
            
        trainoutputs.append(output[0][0])
        
        #error
        error += mse(y,output)
            

        #backward prop
        grad = mse_prime(y,output)
        for layer in reversed(network):
            grad = layer.backward(grad, LEARNING_RATE)

    if (e+1) % 5 == 0 or e == 0:
        plt.cla()
        surf = ax.plot_trisurf(trainingX, trainingY, trainoutputs, cmap="winter")
        ax.set_title("Epoch:%s Training Outputs"% (e+1))
        plt.pause(0.1)

    error /= len(trainInputReshaped)
    errfile.write("%s,%s\n"%(e,error))
    print("%d/%d, error=%f"%(e+1,EPOCHS,error))
   
errfile.close()
print("TRAINING FINISHED. STARTING TESTING")

#TESTING

testInputReshaped = np.reshape(testingInputs, (NUM_DATA_POINTS, 2, 1))
testTruthReshaped = np.reshape(testingTruth, (NUM_DATA_POINTS, 1, 1))

print("Network weights:\n")
print("IH Weights:\n",network[0].weights,"\n")
print("HO Weights:\n",network[2].weights, "\n")

f = open("output.csv","w")
inputCounter = 0

testoutputs = []
for x, y in zip(testInputReshaped,testTruthReshaped):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        testoutputs.append(output[0][0])

        #output data to csv
        f.write("%s,%s,%s,%s\n" % (testingX[inputCounter],testingY[inputCounter],testingTruth[inputCounter],output[0][0]))
        inputCounter +=1
f.close()

#RMSE reporting
MSE = np.square(np.subtract(testingY,testoutputs)).mean()
RMSE = math.sqrt(MSE)

print("TESTING FINISHED\nTest Data RMSE:",RMSE)
plt.ioff()
fig = plt.figure()            
axTest = fig.add_subplot(111, projection="3d")
testsurf = axTest.plot_trisurf(testingX, testingY, testoutputs, cmap="winter")
axTest.set_title("f(x,y)=xy Testing Set")
axTest.set_xlabel("x")
axTest.set_ylabel("y")
axTest.set_zlabel("xy")
axTest.set_zlim3d(-25, 25)
fig.colorbar(testsurf, location = "left")
plt.show()
