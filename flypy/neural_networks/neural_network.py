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
                 nn_architecture,# : list[dict(str)],
                 activation=ReLU,
                 learning_rate: np.float=0.001):
        super().__init__()

        # activation function
        self.activation = activation

        # The gradient vector has a direction and a magnitude. Gradient descent algorithms 
        # multiply the magnitude of the gradient by a scalar known as learning rate (also 
        # sometimes called step size) to determine the next point.
        self.__learning_rate = learning_rate
        self.__inputs = X
        self.__outputs = Y
        self.nn_architecture = nn_architecture
        # initialize weights and biases
        # TODO: 3 here should be dynamically computed based on the number
        #       of neurons on the input layer
        
        num_layers = len(nn_architecture)
        np.random.seed(42)
        param_vals = {} 
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            num_inputs = layer["input_dim"]
            num_outputs = layer["output_dim"]
            param_vals['W' + str(layer_idx)] = 2 * np.random.random((num_outputs, num_inputs)) - 1
            param_vals['b' + str(layer_idx)] = 0.1 * np.random.randn(num_outputs, 1)

        self.__param_vals = param_vals
    

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr):

        Z = np.dot(W_curr, A_prev) + b_curr
        A = self.activation.eval(Z)
        A = 1/(1+np.exp(-Z))
        
        return A, Z

    def feedforward(self, X):
        # Forward Propagation
       
        self.__memory = {}
        params = self.__param_vals
        nn_architecture = self.nn_architecture
        
        A_curr = X
        for idx, _ in enumerate(nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr
            w_curr = params['W' +str(layer_idx)]
            b_curr = params['b' +str(layer_idx)] 

            # 
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, w_curr, b_curr)
            
            self.__memory['A' + str(idx)] = A_prev
            self.__memory['Z' + str(layer_idx)] = Z_curr
            
        return A_curr    
    
    def get_accuracy_value(self, Y_hat: np.ndarray, Y: np.ndarray):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=1).mean()

    # an auxiliary function that converts probability into class
    @classmethod
    def convert_prob_into_class(cls, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    @classmethod
    def cost(cls, Y_hat: np.ndarray, Y: np.ndarray):
        """Cost function is the mean square error (MSE) """
        m = Y_hat.shape[1]
        #J = -(1 / m) * np.sum([y_actual[i] * np.log(ypred[i]) + (1-y_actual[i]) * np.log(1-ypred[i]) for i in range(m)])
        #return J
        m = Y_hat.shape[1]
        #return -np.mean((Y_hat - Y)**2)
        cost = -1 / m * (np.dot(Y.T, np.log(Y_hat).T) + np.dot(1 - Y.T, np.log(1 - Y_hat).T))

        return np.squeeze(cost)

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

    @classmethod
    def sigmoid_backward(cls, dA, Z):
        sig = 1/(1+np.exp(-Z))
        return dA * sig * (1 - sig)

    @classmethod
    def relu_backward(cls, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev):
        
        m = A_prev.shape[1]
        relu_backward = self.relu_backward
        backward_activation_func = self.sigmoid_backward
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        #dZ_curr = self.activation.derivative(Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr        
         
    def full_backward_propagation(self, Y_hat, Y):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
        memory = self.__memory
        params_values = self.__param_vals
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            
            dA_curr = dA_prev
            
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]
            
            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev)
            
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
       
        self.__grads_values = grads_values


    def update(self):
        for layer_idx, _ in enumerate(self.nn_architecture, 1):
            self.__param_vals["W" + str(layer_idx)] -= self.__learning_rate * self.__grads_values["dW" + str(layer_idx)]        
            self.__param_vals["b" + str(layer_idx)] -= self.__learning_rate * self.__grads_values["db" + str(layer_idx)]

    def train(self, X, Y,  epochs: int=10000):
        params_values = self.__param_vals
        cost_history = []
        accuracy_history = []
        
        for i in range(epochs):
            Y_hat = self.feedforward(X)
            cost = self.cost(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            if i % 100 == 0:
                print("Epoch: ", i, ", cost: ", cost, " accuracy:", accuracy)

            self.full_backward_propagation(Y_hat, Y)
            self.update()
            
        return params_values, cost_history, accuracy_history
    #def train_mine(self, batch_size: int=16, epochs: int=10000):
    #    for i in range(epochs):
    #        self.feedforward()
    #        self.backwardpropagation()   
    #        #w, b, cost = self.optimize()
    #       #if i % 100 == 0:
    #        print("Epoch: ", i, " \tcost: ", cost) 