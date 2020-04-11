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

        # The gradient vector has a direction and a magnitude. Gradient descent algorithms 
        # multiply the magnitude of the gradient by a scalar known as learning rate (also 
        # sometimes called step size) to determine the next point.
        self.learning_rate = learning_rate
        
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
   
    def train(self, batch_size: int=16, epochs: int=10000):
        # initialize weights and biases
        # TODO: 3 here should be dynamically computed based on the number
        #       of neurons on the input layer
        
        # Number of output neurons
        n = 2
        
        # Number of input neurons
        m = 4
        self.__biases = np.zeros((n,1))

        # Initialize weights to a random number with a mean of 0.5
        self.__weights = 2 * np.random.random((n,m)) - 1
        
        for i in range(epochs):
            # Forward Propagation
            # NOTE: Might need to transpose inputs here
            z = np.dot(self.__inputs.T, self.__weights.T) + self.__biases
            A = self.activation.eval(z)
            
            # Calculate Loss
            error = self.cost(A, self.__outputs) 
            error = np.squeeze(error) 
            # Backward Propagation

            # Gradient Descent 
            # partial derivatives of weight and bias w.r.t cost
            dw = (1 / m) * np.dot(self.__inputs.T, (A - self.__outputs).T)
            db =  (1 / m) * A - self.__outputs
            
            # Update the weights
            self.__weights -= self.learning_rate * dw
            self.__biases -= self.learning_rate * db
            #if i % 100 == 0:
            print("Epoch: ", i, " \tcost: ", error) 