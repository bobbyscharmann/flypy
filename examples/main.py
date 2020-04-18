from flypy.neural_networks.activation_functions import Sigmoid, ReLU
from flypy.neural_networks import NeuralNetworkTwoLayers
import numpy as np

l = ReLU()
#l.plot(-10, 10, derivative=False)
print("Hello world.")

nn_architecture = [
    {"input_dim": 2, "output_dim": 4},
    {"input_dim": 4, "output_dim": 6},
    {"input_dim": 6, "output_dim": 6},
    {"input_dim": 6, "output_dim": 4},
    {"input_dim": 4, "output_dim": 1},
]
X = np.asarray([[0, 0], [0, 1], [1, 0], [1,1]]).T
Y = np.asarray([[0], [1], [1], [0]])
nn = NeuralNetworkTwoLayers(X=X,
                            Y=Y, 
                            nn_architecture=nn_architecture,
                            activation=Sigmoid)

nn.train(X, Y)
