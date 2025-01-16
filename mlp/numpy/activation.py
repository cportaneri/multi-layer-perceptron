import numpy as np
from abc import ABC, abstractmethod
from mlp.enums import ActivationType

class ActivationNumpy(ABC):
    @abstractmethod
    def type(self):
        pass
    
    @abstractmethod
    def activation(self, layer):
        pass

    @abstractmethod
    def derivative(self, layer):
        pass

class SoftmaxNumpy(ActivationNumpy):
    def type(self):
        return ActivationType.SOFTMAX

    def activation(self, layer):
        max_logit = np.max(layer.weighted_sum, axis=0, keepdims=True)
        exps = np.exp(layer.weighted_sum - max_logit)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def derivative(self, layer):
        return np.tile(layer.error, (1, layer.inputs.shape[1]))
        
class ReluNumpy(ActivationNumpy):
    def type(self):
        return ActivationType.RELU

    def activation(self, layer):
        return np.maximum(0, layer.weighted_sum)

    def derivative(self, layer):
        return layer.error * (layer.weighted_sum > 0)
    
class SigmoidNumpy(ActivationNumpy):
    def type(self):
        return ActivationType.SIGMOID

    def activation(self, layer):
        return 1. / (1. + (np.exp(-layer.weighted_sum)))

    def derivative(self, layer):
        sigmoid = 1. / (1. + np.exp(-layer.weighted_sum))
        return layer.error * sigmoid * (1. - sigmoid)
    
class TanhNumpy(ActivationNumpy):
    def type(self):
        return ActivationType.TANH

    def activation(self, layer):
        return np.tanh(layer.weighted_sum)

    def derivative(self, layer):
        tanh = np.tanh(layer.weighted_sum)
        return layer.error * (1. - tanh*tanh)