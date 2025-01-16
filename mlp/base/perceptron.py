import math
import random
import copy
from mlp.enums import ActivationType

class Perceptron:
    def __init__(self, inputs: list[float], weights: list[float], bias: float, activation) -> None:
        if len(inputs) != len(weights):
            raise ValueError("Inputs and weights do not have the same length.")
        
        if len(inputs) == 0 or len(weights) == 0:
            raise ValueError("Inputs or weights are empty.")
        
        if not isinstance(bias, (int, float)):
            raise ValueError("Bias is not a numeric value.")

        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.value_before_activation = 0.
        self.prediction = 0.
        self.error = 0.
        self.gradient_error = 0.

    def set_optimizer(self, _optimizer):
        self.optimizer = copy.deepcopy(_optimizer)
        self.optimizer.init(self)

    def __str__(self):
        return f"Perceptron with {len(self.inputs)} inputs:\n{self.inputs}\nassociated weights:\n{self.weights}\nbias: {self.bias}\nactivation: {self.activation.type().value}"
    
    def compute_weighted_sum(self):
        self.value_before_activation = self.bias + sum(x * y for x,y in zip(self.inputs, self.weights))
        return self.value_before_activation
    
    def compute_weighted_sum_with_dropout(self, dropout_rate):
        if random.random() < dropout_rate:
            if self.activation.type() == ActivationType.SOFTMAX:
                self.value_before_activation = -math.inf
            else:
                self.value_before_activation = 0.0
        else:
            self.compute_weighted_sum()
            self.value_before_activation *= (1.0 / (1.0 - dropout_rate))
    
    def predict(self):
        self.prediction = self.activation.activation(self.value_before_activation)
        return self.prediction
    
    def compute_gradient(self):
        self.gradient_error = self.error * self.activation.derivative(self.value_before_activation)
        return self.gradient_error