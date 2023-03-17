# NN-WifiCamera

This is a neural network used to process the data received from https://github.com/BrendanBetterman/RSSI-Camera

In this case there are 100 inputs (for a 10 pixel by 10 pixel image) being read from /inputdata/ and their respective folder for their classifications.

In order to change the layout of the neural network, or the layout of the nodes, simply change the network array.

### Example:
```python
network = [Layer(100,20),
    Sigmoid(),
    Layer(20,1)]
```
</example>
This gives a network with 100 inputs, 1 hidden layer with 20 nodes and a sigmoid activation function, with one output.

### Example 2:
```python
network = [Layer(3,25),
    Sigmoid(),
    Layer(25,30),
    Tanh(),
    Layer(30,2)]
```
This gives a network with 3 inputs, 2 hidden layers (one with 25 nodes and a sigmoid activation function, and another with 30 nodes and a tanh activation function), and 2 outputs.

This program will also generate a CSV with the error of each epoch named "error.csv"
