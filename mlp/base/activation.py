import math
from abc import ABC, abstractmethod
from mlp.enums import ActivationType

class Activation(ABC):
    @abstractmethod
    def type(self):
        pass
        
    @abstractmethod
    def activation(self, value):
        pass

    @abstractmethod
    def derivative(self, value):
        pass

    @abstractmethod
    def layer_activation(self, layer):
        pass

class Softmax(Activation):
    def type(self):
        return ActivationType.SOFTMAX

    def activation(self, value):
        return value

    def derivative(self, value):
        return 1.
    
    def layer_activation(self, layer):
        logits = []
        for neuron in layer.neurons:
            logits.append(neuron.value_before_activation)

        max_logit = max(logits)
        exps = [math.exp(max(min(logit - max_logit, 709), -709)) for logit in logits]
        sum_exps = sum(exps)
        softmax_predictions = [exp / sum_exps for exp in exps]
        
        for i, neuron in enumerate(layer.neurons):
            neuron.prediction = softmax_predictions[i]

        return softmax_predictions

class NeuronActivation(Activation):
    def layer_activation(self, layer):
        predictions = []
        for neuron in layer.neurons:
            predictions.append(neuron.predict())
        return predictions
        
class Relu(NeuronActivation):
    def type(self):
        return ActivationType.RELU

    def activation(self, value):
        return max(0, value)

    def derivative(self, value):
        return 1. if value > 0 else 0.
    
class Sigmoid(NeuronActivation):
    def type(self):
        return ActivationType.SIGMOID

    def activation(self, value):
        return 1. / (1. + (math.exp(-value)))

    def derivative(self, value):
        return self.activation(value) * (1. - self.activation(value))
    
class Tanh(NeuronActivation):
    def type(self):
        return ActivationType.TANH

    def activation(self, value):
        return math.tanh(value)

    def derivative(self, value):
        return 1. - (math.tanh(value) * math.tanh(value))