import numpy as np
from abc import ABC, abstractmethod
from mlp.numpy.layer import SingleLayerPerceptronNumpy
from mlp.enums import OptimizerType

class OptimizerNumpy(ABC):
    @abstractmethod
    def type(self):
        pass
    
    @abstractmethod
    def init(self, layer: SingleLayerPerceptronNumpy):
        pass

    @abstractmethod
    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float, **kwargs):
        pass

class StochasticGradientDescentNumpy(OptimizerNumpy):
    def type(self):
        return OptimizerType.SGD
    
    def init(self, layer: SingleLayerPerceptronNumpy):
        self.layer = layer

    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float):
        self.layer.weights -= learning_rate * (np.dot(self.layer.gradient_error, self.layer.inputs.T) + l2_lambda*self.layer.weights)
        self.layer.bias -= (learning_rate * self.layer.gradient_error)

class AdamNumpy(OptimizerNumpy):
    def type(self):
        return OptimizerType.ADAM
    
    def init(self, layer: SingleLayerPerceptronNumpy):
        self.layer = layer
        self.m_weights = np.zeros((layer.layer_size, layer.previous_layer_size))
        self.v_weights = np.zeros((layer.layer_size, layer.previous_layer_size))
        self.m_bias = np.zeros((layer.layer_size, 1))
        self.v_bias = np.zeros((layer.layer_size, 1))
        self.t = 0

    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):    
        self.t += 1

        grad_weights = np.dot(self.layer.gradient_error, self.layer.inputs.T)
        grad_bias = np.sum(self.layer.gradient_error, axis=1, keepdims=True)

        self.m_weights = beta1 * self.m_weights + (1 - beta1) * grad_weights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (grad_weights ** 2)

        m_hat = self.m_weights / (1 - beta1**self.t)
        v_hat = self.v_weights / (1 - beta2**self.t)

        self.layer.weights -= ((learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)) + (learning_rate * l2_lambda * self.layer.weights)
        
        self.m_bias = beta1 * self.m_bias + ((1 - beta1) * grad_bias)
        self.v_bias = beta2 * self.v_bias + ((1 - beta2) * (grad_bias**2))

        m_hat_bias = self.m_bias / (1 - beta1**self.t)
        v_hat_bias = self.v_bias / (1 - beta2**self.t)

        self.layer.bias -= (learning_rate * m_hat_bias) / (np.sqrt(v_hat_bias) + epsilon)