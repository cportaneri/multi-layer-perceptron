from mlp.enums import MLPType, ActivationType
from mlp.base.layer import SingleLayerPerceptron

class MultiLayerPerceptron:
    def __init__(self) -> None:
        self.layers = []

    def type(self):
        return MLPType.BASE
    
    def depth(self) -> int:
        return len(self.layers)
    
    def input_dim(self) -> int:
        return self.layers[0].previous_layer_size
    
    def output_dim(self) -> int:
        return self.layers[-1].dim()
    
    def get_layers(self):
        return self.layers

    def add_layer(self, layer_dim, activation, dropout_rate, input_dim):
        if layer_dim <= 0:
            raise ValueError("Layer size must be greater than zero.")
        
        if not (0 <= dropout_rate < 1):
            raise ValueError("Dropout rate must be in the range [0, 1).")

        if input_dim == -1: 
            if len(self.layers) < 1:
                raise ValueError("input_dim must be set for the first layer")
            else:
                input_dim = self.layers[-1].dim()
        
        self.layers.append(SingleLayerPerceptron(layer_dim, input_dim, activation, dropout_rate))

    def set_training_mode(self, _training_mode):
        for layer in self.layers:
            layer.training_mode = _training_mode

    def set_input_values(self, input_values):
        self.layers[0].set_input_values(input_values)

    def set_optimizer(self, _optimizer):
         self.optimizer = _optimizer
         for layer in self.layers:
            layer.set_optimizer(_optimizer)

    def set_loss_function(self, _loss_function):
        self.loss_function = _loss_function

    def propagate(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].propagate(self.layers[i+1])

    def predict(self):
        self.propagate()
        return self.layers[-1].predict()

    def batch_prediction(self, batch_input, batch_output, batch_size, l2_lambda):
        batch_loss = 0
        batch_error_vector = [0] * self.output_dim()
        for input, target_output in zip(batch_input, batch_output):

            self.set_input_values(input)
            prediction = self.predict()

            loss = self.loss_function.compute(prediction, target_output)
            batch_loss += self.l2_regularization_loss(loss, l2_lambda)

            raw_error_vector = self.compute_raw_error(prediction, target_output)
            batch_error_vector = [batch_error_vector[i] + raw_error_vector[i] for i in range(len(raw_error_vector))]

        batch_error_vector = [raw_error / batch_size for raw_error in batch_error_vector]
        batch_loss /= batch_size
        return (batch_error_vector, batch_loss)
    
    def set_errors(self, errors):
        if len(errors) != self.layers[-1].dim():
            raise ValueError(f"Expected {self.layers[-1].dim()} errors, got {len(errors)}.")
        
        self.layers[-1].set_errors(errors)

    def back_propagate(self):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].back_propagate(self.layers[i-1])
        self.layers[0].compute_gradients()
    
    def update_weights_and_biases(self, raw_errors, learning_rate: float, l2_lambda: float):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        
        self.set_errors(raw_errors)
        self.back_propagate()
        for layer in self.layers:
            layer.update_weights_and_biases(learning_rate, l2_lambda)

    def l2_regularization_loss(self, loss, l2_lambda):
        l2_penalty = 0.
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    l2_penalty += (weight * weight)

        l2_penalty = l2_penalty * 0.5 * l2_lambda
        return loss + l2_penalty

    def compute_raw_error(self, prediction, target_output):
        return [prediction[i] - target_output[i] for i in range(len(prediction))]