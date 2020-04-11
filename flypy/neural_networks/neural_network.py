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
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 num_layers: int,
                 activation=ReLU,
                 learning_rate: np.float=0.001):
        super().__init__()

        # activation function
        self.activation = activation

        # The gradient vector has a direction and a magnitude. Gradient descent algorithms 
        # multiply the magnitude of the gradient by a scalar known as learning rate (also 
        # sometimes called step size) to determine the next point.
        self.learning_rate = learning_rate
        self.__inputs = X
        self.__outputs = Y
        self.num_layers = num_layers
        # initialize weights and biases
        # TODO: 3 here should be dynamically computed based on the number
        #       of neurons on the input layer
        
        # Number of layers
        n = num_layers
        
        # Number of input neurons
        m = 4
        self.__biases = np.zeros((num_layers,m))

        # Initialize weights to a random number with a mean of 0.5
        self.__weights1 = 2 * np.random.random((m, num_layers)) - 1
        self.__weights2 = 2 * np.random.random((num_layers, 1)) - 1
        
         
    @classmethod
    def cost(cls, ypred: np.ndarray, y_actual: np.ndarray):
       """Cost function is the mean square error (MSE) """
       m = ypred.shape[1]
       J = -(1 / m) * np.sum([y_actual[i] * np.log10(ypred[i]) + (1-y_actual[i]) * np.log10(1-ypred[i]) for i in range(m)])
       return J
       #return np.sqrt(ypred**2 + y_actual**2)
   
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
  
    def forwardpropagation(self):
        # Forward Propagation
        # NOTE: Might need to transpose inputs here
        layer1_z = np.dot(self.__inputs.T, self.__weights1) + self.__biases
        self.layer1 = self.activation.eval(layer1_z)
        layer2_z = np.dot(self.layer1, self.__weights2) + self.__biases
        self.y_hat = self.activation.eval(layer2_z)
        
    def backwardpropagation(self):    
        # Calculate Loss
        d_weights2 = self.cost(self.y_hat, self.__outputs) 
        d_weights1 = self.cost(self.layer1, self.__outputs) 
        error = np.squeeze(cost) 
        # Backward Propagation

        # Gradient Descent 
        # partial derivatives of weight and bias w.r.t cost
        dw = (1 / m) * np.dot(self.__inputs.T, (A - self.__outputs).T)
        db =  (1 / m) * A - self.__outputs
        
        # Update the weights
        self.__weights -= self.learning_rate * dw
        self.__biases -= self.learning_rate * db

    def train(self, batch_size: int=16, epochs: int=10000):
        w = self.__weights
        b = self.__biases
        for i in range(epochs):
            self.forwardpropagation(w, b)
            self.backwardpropagation()   
            w, b, cost = self.optimize()
           #if i % 100 == 0:
            print("Epoch: ", i, " \tcost: ", cost) 