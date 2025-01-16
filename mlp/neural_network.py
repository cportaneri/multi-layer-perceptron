import copy
import time
from mlp.utils import shuffle_data
from mlp.enums import MetricType
from mlp.factory import Factory

class NeuralNetwork:
    def __init__(self, mlp_type) -> None:
        self.mlp = Factory.create_mlp(mlp_type)

    def reset(self):
        self.__init__(self.mlp)

    def set_training_data(self, training_dataset_inputs: list, training_dataset_outputs: list) -> None:
        if not training_dataset_inputs or not training_dataset_outputs:
            raise ValueError("Training data input or output is empty.")

        if len(training_dataset_inputs) != len(training_dataset_outputs):
            raise ValueError("Training data input and output have not the same size.")
        
        self.training_data_inputs, self.training_data_outputs = shuffle_data(training_dataset_inputs, training_dataset_outputs)

    def set_training_mode(self, training_mode: bool):
        self.training_mode = training_mode
        self.mlp.set_training_mode(training_mode)

    def set_hyper_parameters(self, learning_rate, batch_size, max_epoch, loss_function_type, optimizer_type, l2_lambda: float = 0.):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mlp.batch_size = batch_size
        self.max_epoch = max_epoch
        self.l2_lambda = l2_lambda
        self.optimizer_type = optimizer_type
        self.loss_function_type = loss_function_type
        self.mlp.set_optimizer(Factory.create_optimizer(self.mlp.type(), optimizer_type))
        self.mlp.set_loss_function(Factory.create_metric(self.mlp.type(), loss_function_type))

    def add_layer(self, layer_dim, activation_type, dropout_rate=0., input_dim = -1):
        if layer_dim <= 0:
            raise ValueError("Layer size must be greater than zero.")
        
        if input_dim == -1 and self.mlp.depth() < 1:
            raise ValueError("input_dim must be set for the first layer")
        
        activation = Factory.create_activation(self.mlp.type(), activation_type)
        self.mlp.add_layer(layer_dim, activation, dropout_rate, input_dim)

    def print_configuration(self):
        if self.mlp.depth() < 2:
            raise ValueError("Network must have at least two layers: one hidden and one output")
                
        print("Neural Network Configuration:")
        print(f" Input dimension: {self.mlp.input_dim()}")
        print(f" Depth: {self.mlp.depth()}")
        for i, layer in enumerate(self.mlp.get_layers()):
            print(f"  Layer {i} dimension: {layer.dim()}, activation : {layer.activation.type().value}")
        print(f" Learning Rate: {self.learning_rate}")
        print(f" Max Epochs: {self.max_epoch}")
        print(f" Batch Size: {self.batch_size}")
        print(f" Optimizer: {self.optimizer_type.value}")
        print(f" Error Loss: {self.loss_function_type.value}")
        print(f" L2 Regularization Lambda: {self.l2_lambda}")

    def predict(self, input):
        self.mlp.set_input_values(input)
        return self.mlp.predict()

    def train(self):
        if self.mlp.depth() < 2:
            raise ValueError("Network must have at least two layers: one hidden and one output")

        best_epoch = 0
        best_loss = float('inf')
        best_model = None
        patience = 1 # can be a parameter
        patience_counter = 0

        self.set_training_mode(True)
        start_time = time.time()
        for epoch in range(self.max_epoch):
            epoch_loss = 0
            
            self.training_data_inputs, self.training_data_outputs = shuffle_data(self.training_data_inputs, self.training_data_outputs)
            num_batches = len(self.training_data_inputs) // self.batch_size
            for batch_index in range(num_batches):

                batch_input = self.training_data_inputs[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
                batch_output = self.training_data_outputs[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]

                (batch_error_vector, batch_loss) = self.mlp.batch_prediction(batch_input, batch_output, self.batch_size, self.l2_lambda)
                self.mlp.update_weights_and_biases(batch_error_vector, self.learning_rate, self.l2_lambda)

                epoch_loss += batch_loss
            
            epoch_loss /= num_batches
            print(f"Epoch #{epoch} Loss: {epoch_loss}")
            
            current_model_copy = copy.deepcopy(self.mlp)
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model = current_model_copy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch #{epoch} due to overfitting")
                print(f"Best epoch is #{best_epoch}")
                break

        if best_model:
            self.mlp = best_model

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

    def validation(self, validation_type, **kwargs):
        validation = Factory.create_validation(validation_type)
        return validation.validation(self, **kwargs)

    def test(self, dataset_inputs, dataset_outputs, metric_types: list[MetricType]):
        predictions = []
        
        self.set_training_mode(False)
        start_time = time.time()
        for input in dataset_inputs:
            predictions.append(self.predict(input))
        end_time = time.time()
        testing_time = end_time - start_time

        results = []
        for metric_type in metric_types:
            metric = Factory.create_metric(self.mlp.type(), metric_type)
            metric_result = metric.compute_average(predictions, dataset_outputs)
            results.append(metric_result)

        print(f"Testing time: {testing_time:.2f} seconds")
        return results