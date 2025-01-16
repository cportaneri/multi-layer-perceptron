import random
from itertools import product
from abc import ABC, abstractmethod
from mlp.enums import MLPType
from mlp.metric import Accuracy, Precision, Recall, F1Score
from mlp.base.loss import MeanSquaredError, CrossEntropy
from mlp.numpy.loss import MeanSquaredErrorNumpy, CrossEntropyNumpy
from mlp.base.optimizer import StochasticGradientDescent, Adam
from mlp.numpy.optimizer import StochasticGradientDescentNumpy, AdamNumpy
from mlp.base.activation import Relu, Softmax, Sigmoid, Tanh
from mlp.numpy.activation import ReluNumpy, SoftmaxNumpy, SigmoidNumpy, TanhNumpy
        
class HyperparametersTuning(ABC):
    def hyperparameters_tuning(self, neural_network, input_dim, output_dim, tuning_metric, validation, **kwargs):
        hyperparameter_space = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 48, 64],
            'depth': [3, 4, 5],
            'layer_size': [16, 24, 32],
            'activation_type': [
                [Relu(), Sigmoid(), Tanh()] 
                if neural_network.mlp.type() == MLPType.NUMPY
                else [ReluNumpy(), SigmoidNumpy(), TanhNumpy()] 
            ],
            'optimizer_type': [
                [StochasticGradientDescentNumpy(), AdamNumpy()] 
                if neural_network.mlp.type() == MLPType.NUMPY
                else [StochasticGradientDescent(), Adam()]
            ],
            'max_epoch': [5, 7, 10],
            'l2_lambda': [0.0, 0.0001, 0.0002],
            'error_loss': (
                [MeanSquaredErrorNumpy(), CrossEntropyNumpy()] 
                if neural_network.mlp.type() == MLPType.NUMPY
                else [MeanSquaredError(), CrossEntropy()]
            ),
        }

        best_hyperparameters, best_validation_loss = self.search(neural_network, hyperparameter_space, input_dim, output_dim, tuning_metric, validation, **kwargs)

        print("Best Hyperparameters:", best_hyperparameters)
        print("Best Validation Loss:", best_validation_loss)

        neural_network.reset()
        neural_network.add_layer(best_hyperparameters['layer_size'], best_hyperparameters['activation_type'], input_dim)
        for _ in range(best_hyperparameters['depth'] - 2):
            neural_network.add_layer(best_hyperparameters['layer_size'], best_hyperparameters['activation_type'])

        if neural_network.mlp.type() == MLPType.NUMPY:
            neural_network.add_layer(output_dim, SoftmaxNumpy())
        else : neural_network.add_layer(output_dim, Softmax())
        
        error_metrics = [Accuracy(), Precision(), Recall(), F1Score()]
        results = validation.validation(
            neural_network, best_hyperparameters['learning_rate'], best_hyperparameters['batch_size'],
            best_hyperparameters['max_epoch'], best_hyperparameters['optimizer_type'],
            best_hyperparameters['l2_lambda'], best_hyperparameters['error_loss'], error_metrics, **kwargs
        )

        print("Results:")
        print(f"Accuracy: {results[0]:.2f}%")
        print(f"Precision: {results[1]:.2f}%")
        print(f"Recall: {results[2]:.2f}%")
        print(f"F1-Score: {results[3]:.2f}%")
        self.print_configuration()

        # Return the best hyperparameters, best validation loss, and final evaluation results
        return {
            "best_hyperparameters": best_hyperparameters,
            "best_validation_loss": best_validation_loss,
            "validation_results": results
        }
    
    @abstractmethod
    def search(self, hyperparameter_space, input_dim, output_dim, tuning_metric, validation_type, **kwargs):
        pass

class GridSearch(HyperparametersTuning):
    def search(self, neural_network, hyperparameter_space, input_dim, output_dim, tuning_metric, validation, **kwargs):
        from mlp.base.activation import Softmax
        from mlp.numpy.activation import SoftmaxNumpy
        best_hyperparameters = None
        best_validation_loss = float('inf')

        hyperparameter_combinations = list(product(*hyperparameter_space.values()))
        for combination in hyperparameter_combinations:
            hyperparameters = dict(zip(hyperparameter_space.keys(), combination))

            neural_network.reset()
            neural_network.add_layer(hyperparameters['layer_size'], hyperparameters['activation_type'], input_dim)
            for _ in range(hyperparameters['depth'] - 2):
                neural_network.add_layer(hyperparameters['layer_size'], hyperparameters['activation_type'])

            if neural_network.mlp.type() == MLPType.NUMPY:
                neural_network.add_layer(output_dim, SoftmaxNumpy())
            else : neural_network.add_layer(output_dim, Softmax())

            validation_loss = validation.validation(
                neural_network, hyperparameters['learning_rate'], hyperparameters['max_epoch'],
                hyperparameters['batch_size'], hyperparameters['l2_lambda'], hyperparameters['optimizer_type'],
                hyperparameters['error_loss'], tuning_metric, **kwargs
            )[0]

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_hyperparameters = hyperparameters

        return best_hyperparameters, best_validation_loss

class RandomSearch(HyperparametersTuning):
    def search(self, neural_network, hyperparameter_space, input_dim, output_dim, tuning_metric, validation, n_iterations, **kwargs):
        from mlp.base.activation import Softmax
        from mlp.numpy.activation import SoftmaxNumpy
        best_hyperparameters = None
        best_validation_loss = float('inf')

        for _ in range(n_iterations):
            hyperparameters = {key: random.choice(values) for key, values in hyperparameter_space.items()}

            neural_network.reset()
            neural_network.add_layer(hyperparameters['layer_size'], hyperparameters['activation_type'], input_dim)
            for _ in range(hyperparameters['depth'] - 2):
                neural_network.add_layer(hyperparameters['layer_size'], hyperparameters['activation_type'])

            if neural_network.mlp.type() == MLPType.NUMPY:
                neural_network.add_layer(output_dim, SoftmaxNumpy())
            else : neural_network.add_layer(output_dim, Softmax())

            validation_loss = validation.validation(
                neural_network, hyperparameters['learning_rate'], hyperparameters['max_epoch'],
                hyperparameters['batch_size'], hyperparameters['l2_lambda'], hyperparameters['optimizer_type'],
                hyperparameters['error_loss'], tuning_metric, **kwargs
            )[0]

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_hyperparameters = hyperparameters

        return best_hyperparameters, best_validation_loss