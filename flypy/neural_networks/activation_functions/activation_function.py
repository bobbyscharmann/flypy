"""Abstract class for implementing an activation function
Reference: https://en.wikipedia.org/wiki/Activation_function

In artificial neural networks, the activation function of a node defines the 
output of that node given an input or set of inputs. In biologically inspired 
neural networks, the activation function is usually an abstraction representing
the rate of action potential firing in the cell.[2] In its simplest form, 
this function is binary—that is, either the neuron is firing or not.
 
Nonlinear – When the activation function is non-linear, then a two-layer neural 
            network can be proven to be a universal function approximator.[6] 
            The identity activation function does not satisfy this property. 
            When multiple layers use the identity activation function, the 
            entire network is equivalent to a single-layer model.
Range – When the range of the activation function is finite, gradient-based 
        training methods tend to be more stable, because pattern presentations
        significantly affect only limited weights. When the range is infinite,
        training is generally more efficient because pattern presentations 
        significantly affect most of the weights. In the latter case, smaller
        learning rates are typically necessary.[citation needed]
Continuously differentiable – This property is desirable (RELU is not continuously 
                              differentiable and has some issues with gradient-based 
                              optimization, but it is still possible) for enabling 
                              gradient-based optimization methods. The binary step 
                              activation function is not differentiable at 0, and it 
                              differentiates to 0 for all other values, so gradient-based
                               methods can make no progress with it.[7]
Monotonic – When the activation function is monotonic, the error surface associated with 
            a single-layer model is guaranteed to be convex.[8]
Smooth functions with a monotonic derivative – These have been shown to generalize better in some cases.
Approximates identity near the origin – When activation functions have this property, 
                                        the neural network will learn efficiently when
                                         its weights are initialized with small random 
                                         values. When the activation function does not 
                                         approximate identity near the origin, special 
                                         care must be used when initializing the weights.
 
"""
from abc import ABC
from abc import abstractmethod
import numpy as np
from plotly import graph_objects as go

"""Class implementing the interface for an activation function."""
class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()
        
    @classmethod
    @abstractmethod 
    def eval(cls, x: np.float) -> np.float:
        pass
    
    @classmethod
    @abstractmethod 
    def derivative(cls, x: np.float) -> np.float:
        pass
    
    def plot(self, xmin: np.float, xmax: np.float, derivative: bool=False) -> None:
        x_vals = np.linspace(start=xmin, 
                             stop=xmax,
                             num=200)
        y_vals = self.eval(x_vals) if not derivative else self.derivative(x_vals)
        fig = go.Figure(data=go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers'
        ))
        fig.update_layout(
            title='Logistic Activation Function ' if not derivative else 
            "Derivative for the Logistic Activation Function",
            title_x=0.5,
            xaxis_title='Value',
            yaxis_title='logistic(value)',
        )        
        fig.show() 