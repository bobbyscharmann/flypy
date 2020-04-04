"""Abstract class for implementing an activation function"""
from abc import ABC
from abc import abstractmethod
import numpy as np


"""Class implementing the interface for an activation function."""
class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod 
    def eval(self, x: np.float) -> np.float:
        pass
    
    @abstractmethod 
    def derivative(self, x: np.float) -> np.float:
        pass