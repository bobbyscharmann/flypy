from typing import ClassVar
from flypy.neural_networks.activation_functions import ActivationFunction, ReLU, Sigmoid
import numpy as np

class NeuralNetworkTwoLayers(object):
    """Defines a neural network""" 
    __inputs: np.ndarray = []
    __outputs: np.ndarray = []

    # Neural Network weights and biases
    __weights:np.ndarray = []
    __biases:np.ndarray = [] 
    
    # NOTE: Not doing type hinting with the activation function due to apparent limitations
    # with mypy
    def __init__(self, activation=ReLU,
                learning_rate: np.float=0.001):
        super().__init__()

        # activation function
        self.activation = activation
        self.learning_rate = learning_rate
        
    @classmethod
    def cost(cls, ypred: np.float, y_actual: np.float):
       """Cost function is the mean square error (MSE) """
       
       return np.sqrt(ypred**2 + y_actual**2)
   
    @property
    def inputs(self) -> None:
        return self.__inputs
    
    @inputs.setter
    def inputs(self, value: np.ndarray) -> None:
        self.__inputs = value

    @property
    def outputs(self) -> None:
        return self.__outputs
    
    @outputs.setter
    def outputs(self, value: np.ndarray) -> None:
        self.__outputs = value
   
    def train(self, batch_size: int=16, epochs: int=10000):
        # initialize weights and biases
        # TODO: 3 here should be dynamically computed based on the number
        #       of neurons on the input layer
        self.__biases = np.zeros((3,1))

        # Initialize weights to a random number with a mean of 0.5
        self.__weights = 2 * np.random.random((3,1)) - 1
        
        # Forward Propagation
        z = np.dot(self.inputs, self.__weights) + self.__biases
        a = self.activation.eval(z)
        
        # Calculate Loss
        error = self.cost(a, self.outputs) 
        
        # Backward Propagation
        
        # Gradient Descent 