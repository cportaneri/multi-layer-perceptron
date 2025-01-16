import numpy as np
import copy
from mlp.enums import MLPType, ActivationType, WeightsInitType

class SingleLayerPerceptronNumpy:
    def __init__(self, layer_size, previous_layer_size, activation, dropout_rate=0.) -> None:
        self.layer_size = layer_size
        self.previous_layer_size = previous_layer_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.weights = self.weights_init()
        self.bias = self.bias_init()

    def dim(self) -> int:
        return self.layer_size

    def set_input_values(self, input_values):
        if not isinstance(input_values, np.ndarray):
            input_values = np.array([input_values]).T
            
        if input_values.shape[0] != self.previous_layer_size:
           raise ValueError(f"Expected {self.previous_layer_size} input values, "
                            f"but got {input_values.shape[0]}.")

        self.inputs = input_values

    def set_optimizer(self, _optimizer):
        self.optimizer = copy.deepcopy(_optimizer)
        self.optimizer.init(self)

    def weights_init(self):
        from mlp.factory import Factory
        if self.activation.type() == ActivationType.RELU:
            he_normal = Factory.create_weights_init(MLPType.NUMPY, WeightsInitType.HNORM)
            return he_normal.init(self)
        else:
            xavier_normal = Factory.create_weights_init(MLPType.NUMPY, WeightsInitType.XNORM)
            return xavier_normal.init(self)
        
    def bias_init(self):
        return np.full((self.layer_size, 1), 0.)
    
    def propagate(self, next_layer: 'SingleLayerPerceptronNumpy'):
        predictions = self.predict()
        next_layer.set_input_values(predictions)

    def compute_weighted_sum(self):
        self.weighted_sum = np.dot(self.weights, self.inputs) + self.bias
        return self.weighted_sum
    
    def compute_weighted_sum_with_dropout(self):
        self.compute_weighted_sum()
        dropout_mask = (np.random.rand(*self.weighted_sum.shape) > self.dropout_rate)
        if self.activation.type() == ActivationType.SOFTMAX:
            self.weighted_sum = np.where(dropout_mask, self.weighted_sum, -np.inf)
        else:
            self.weighted_sum *= dropout_mask
            self.weighted_sum /= (1.0 - self.dropout_rate)

    def predict(self):
        if self.training_mode:
            self.compute_weighted_sum_with_dropout()
        else:
            self.compute_weighted_sum()

        self.predictions = self.activation.activation(self)
        return self.predictions
       
    def set_errors(self, errors):
        if len(errors) != self.layer_size:
            raise ValueError("Error size not the same as neurons.")

        self.error = errors

    def compute_gradients(self):
        self.gradient_error = self.activation.derivative(self)
        return self.gradient_error

    def back_propagate(self, previous_layer: 'SingleLayerPerceptronNumpy'):
        if self.weights.shape[1] != previous_layer.layer_size:
            raise ValueError("Shape mismatch between layers during backpropagation.")

        previous_layer_errors = np.dot(self.weights.T, self.compute_gradients())
        previous_layer.set_errors(np.mean(previous_layer_errors, axis=1, keepdims=True))