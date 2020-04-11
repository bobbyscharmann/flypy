from flypy.neural_networks.activation_functions import Sigmoid, ReLU
from flypy.neural_networks import NeuralNetworkTwoLayers
import numpy as np

l = ReLU()
#l.plot(-10, 10, derivative=False)
print("Hello world.")

nn = NeuralNetworkTwoLayers(X=np.asarray([[0,0],[0,1],[1,0],[1,1]]),
                            Y=np.asarray([0,1,1,1]), 
                            activation=ReLU,
                            num_layers=2)

nn.train()
