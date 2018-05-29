from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


#  XOR Operation
#   x   y   x XOR y
#   0   0       0
#   1   0       1




neuralNetwork = buildNetwork(2,3,1) # 2 neurons in input, 3 hidden , 1 output

dataSet = SupervisedDataSet(2,1) # input 2d , output 1d

dataSet.addSample((0,0),(0,))
dataSet.addSample((1,0),(1,))
dataSet.addSample((0,1),(1,))
dataSet.addSample((1,1),(0,))

trainer = BackpropTrainer(neuralNetwork,dataSet)

for i in range(1,10000):
    trainer.train(10000)
    if i % 1000 == 0:
        print (neuralNetwork.activate([0,0]))
        print (neuralNetwork.activate([1,0]))
        print (neuralNetwork.activate([0,1]))
        print (neuralNetwork.activate([1,1]))  ## every thousand iteration predict
