from flypy.neural_networks.activation_functions import Sigmoid, ReLU
from flypy.neural_networks import NeuralNetworkTwoLayers
import numpy as np

l = ReLU()
#l.plot(-10, 10, derivative=False)
print("Hello world.")

nn = NeuralNetworkTwoLayers()

nn.inputs = np.asarray([[0,0],[0,1],[1,0],[1,1]])
nn.outputs = np.asarray([0,1,1,1])

nn.train()
