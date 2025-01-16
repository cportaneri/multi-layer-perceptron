from mlp.enums import MLPType, ActivationType, WeightsInitType
from mlp.base.perceptron import Perceptron

class SingleLayerPerceptron:
    def __init__(self, layer_size, previous_layer_size, activation, dropout_rate=0.) -> None:
        self.layer_size = layer_size
        self.previous_layer_size = previous_layer_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.neurons = [Perceptron([-1.0] * self.previous_layer_size, self.weights_init(), self.bias_init(), self.activation) 
                        for _ in range(self.layer_size)]

    def dim(self) -> int:
        return len(self.neurons)

    def set_input_values(self, input_values: list[float]):
        if len(input_values) != self.previous_layer_size:
            raise ValueError(f"Expected input size {self.previous_layer_size}, got {len(input_values)}.")
        
        for i, neuron in enumerate(self.neurons):
            if len(neuron.inputs) != len(input_values):
                raise ValueError(f"Neuron {i} inputs not initialized correctly.")
           
            neuron.inputs = input_values

    def set_optimizer(self, _optimizer):
         for neuron in self.neurons:
            neuron.set_optimizer(_optimizer)

    def weights_init(self):
        from mlp.factory import Factory
        if self.activation.type() == ActivationType.RELU:
            he_normal = Factory.create_weights_init(MLPType.BASE, WeightsInitType.HNORM)
            return he_normal.init(self)
        else:
            xavier_normal = Factory.create_weights_init(MLPType.BASE, WeightsInitType.XNORM)
            return xavier_normal.init(self)
    
    def bias_init(self):
        return 0.01

    def propagate(self, next_layer: 'SingleLayerPerceptron'):
        predictions = self.predict()
        next_layer.set_input_values(predictions)

    def predict(self):
        for neuron in self.neurons:
            if self.training_mode:
                neuron.compute_weighted_sum_with_dropout(self.dropout_rate)
            else:
                neuron.compute_weighted_sum()

        return self.activation.layer_activation(self)

    def set_errors(self, errors: list[float]):
        if len(errors) != len(self.neurons):
            raise ValueError("Error size not the same as neurons.")

        for i, neuron in enumerate(self.neurons):
            neuron.error = errors[i]

    def compute_gradients(self):
        return [neuron.compute_gradient() for neuron in self.neurons]

    def back_propagate(self, previous_layer: 'SingleLayerPerceptron'):
        gradients = self.compute_gradients()
        previous_layer_errors = [sum(gradients[j] * self.neurons[j].weights[i] for j in range(len(self.neurons))) for i in range(len(previous_layer.neurons))]
        previous_layer.set_errors(previous_layer_errors)

    def update_weights_and_biases(self, learning_rate: float, l2_lambda: float):
        for neuron in self.neurons:
            neuron.optimizer.update_weights_and_biases(learning_rate, l2_lambda)