import math
from abc import ABC, abstractmethod
from mlp.base.perceptron import Perceptron
from mlp.enums import OptimizerType

class Optimizer(ABC):
    @abstractmethod
    def type(self):
        pass
    
    @abstractmethod
    def init(self, neuron: Perceptron):
        pass
        
    @abstractmethod
    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float, **kwargs):
        pass

class StochasticGradientDescent(Optimizer):
    def type(self):
        return OptimizerType.SGD
    
    def init(self, neuron: Perceptron):
        self.neuron = neuron

    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float):
        for i in range(len(self.neuron.weights)):
            self.neuron.weights[i] -= learning_rate * (self.neuron.gradient_error * self.neuron.inputs[i] + l2_lambda * self.neuron.weights[i])

        self.neuron.bias -= (learning_rate * self.neuron.gradient_error)

class Adam(Optimizer):
    def type(self):
        return OptimizerType.ADAM
    
    def init(self, neuron: Perceptron):
        self.neuron = neuron
        self.m_weights = [0.0] * len(neuron.weights)
        self.v_weights = [0.0] * len(neuron.weights)
        self.m_bias = 0.
        self.v_bias = 0.
        self.t = 0

    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        self.t += 1

        for i in range(len(self.neuron.weights)):
            self.m_weights[i] = beta1 * self.m_weights[i] + ((1 - beta1) * (self.neuron.gradient_error * self.neuron.inputs[i]))
            self.v_weights[i] = beta2 * self.v_weights[i] + ((1 - beta2) * (self.neuron.gradient_error * self.neuron.inputs[i])**2)

            m_hat = self.m_weights[i] / (1 - beta1**self.t)
            v_hat = self.v_weights[i] / (1 - beta2**self.t)

            self.neuron.weights[i] -= (learning_rate * m_hat) / (math.sqrt(v_hat) + epsilon) + learning_rate * l2_lambda * self.neuron.weights[i]

        self.m_bias = beta1 * self.m_bias + ((1 - beta1) * self.neuron.gradient_error)
        self.v_bias = beta2 * self.v_bias + ((1 - beta2) * (self.neuron.gradient_error)*(self.neuron.gradient_error))

        m_hat_bias = self.m_bias / (1 - beta1**self.t)
        v_hat_bias = self.v_bias / (1 - beta2**self.t)

        self.neuron.bias -= (learning_rate * m_hat_bias) / (math.sqrt(v_hat_bias) + epsilon)