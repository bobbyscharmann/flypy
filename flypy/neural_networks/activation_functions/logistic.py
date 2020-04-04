"""Implementation of the Logistic logistic function
    Reference: https://en.wikipedia.org/wiki/Logistic_function
    
    Logistic functions are often used in neural networks to introduce 
    nonlinearity in the model or to clamp signals to within a specified 
    range. A popular neural net element computes a linear combination of 
    its input signals, and applies a bounded logistic function to the result; 
    this model can be seen as a "smoothed" variant of the classical threshold
    neuron.


"""    
    
from .activation_function import ActivationFunction
import numpy as np

class Logistic(ActivationFunction):
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def eval(cls, x: np.float) -> np.float:
        """Evaluates the logistic function for a given x value
           Reference: https://en.wikipedia.org/wiki/Logistic_function#Derivative
        """
        return 1 / (1 + np.exp(-x))
    
    @classmethod
    def derivative(cls, x: np.float) -> np.float:
        """Evaluates the derivative of the logitistic function for a given x value
           Reference: https://en.wikipedia.org/wiki/Logistic_function#Derivative
        """
        return cls.eval(x) * (1 - cls.eval(x))    
