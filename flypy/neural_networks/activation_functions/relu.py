"""Implementation of the Rectified Linear Unit (ReLU) function
    Reference: 
    https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    
    The rectified linear activation function is a piecewise linear function 
    that will output the input directly if is positive, otherwise, it will 
    output zero. It has become the default activation function for many types
    of neural networks because a model that uses it is easier to train and
    often achieves better performance.




"""    
    
from .activation_function import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def eval(cls, x: np.float) -> np.float:
        """Evaluates the ReLU function for a given x value
           Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        """
        return np.maximum(0.0, x)
    
    @classmethod
    def derivative(cls, x: np.float) -> np.float:
        """Evaluates the derivative of the ReLU function for a given x value
           Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        """
        
        return np.asarray(x>0).astype(np.float)
