from abc import ABC, abstractmethod
from mlp.utils import shuffle_data

class Validation(ABC):
    @abstractmethod
    def validation(self, neural_network, learning_rate, max_epoch, batch_size, l2_lambda, loss_function_type, optimizer_type, metrics,  **kwargs):
        pass

class HoldOut(Validation):
    def validation(self, neural_network, learning_rate, max_epoch, batch_size, l2_lambda, loss_function_type, optimizer_type, metrics, 
                   training_dataset_inputs, training_dataset_outputs, testing_dataset_inputs, testing_dataset_outputs):
        
        print("Initializating neural network...")
        neural_network.set_training_data(training_dataset_inputs, training_dataset_outputs)
        neural_network.set_hyper_parameters(learning_rate, batch_size, max_epoch, loss_function_type, optimizer_type, l2_lambda)
        neural_network.print_configuration()
        
        print("Training neural network...")
        neural_network.train()

        return neural_network.test(testing_dataset_inputs, testing_dataset_outputs, metrics)

class KFold(Validation):
    def validation(self, neural_network, learning_rate, max_epoch, batch_size, l2_lambda, loss_function_type, optimizer_type, metrics, dataset_inputs, dataset_outputs, k):

        dataset_inputs, dataset_outputs = shuffle_data(dataset_inputs, dataset_outputs)

        fold_size = len(dataset_inputs) // k
        folds = [
            (dataset_inputs[i * fold_size:(i + 1) * fold_size], dataset_outputs[i * fold_size:(i + 1) * fold_size])
            for i in range(k)
        ]

        if len(dataset_inputs) % k != 0:
            extra_inputs = dataset_inputs[k * fold_size:]
            extra_outputs = dataset_outputs[k * fold_size:]
            folds[-1] = (folds[-1][0] + extra_inputs, folds[-1][1] + extra_outputs)

        fold_metrics = {metric: [] for metric in metrics}

        for fold_index in range(k):
            print(f"\n=== Fold {fold_index + 1}/{k} ===")

            testing_dataset_inputs, testing_dataset_outputs = folds[fold_index]
            training_dataset_inputs = []
            training_dataset_outputs = []
            for i in range(k):
                if i != fold_index:
                    training_dataset_inputs.extend(folds[i][0])
                    training_dataset_outputs.extend(folds[i][1])

            fold_results = HoldOut().validation(neural_network, learning_rate, max_epoch, batch_size, l2_lambda, loss_function_type, optimizer_type, metrics,
                                                training_dataset_inputs, training_dataset_outputs, testing_dataset_inputs, testing_dataset_outputs)
        
            for metric, value in zip(metrics, fold_results):
                fold_metrics[metric].append(value)

        avg_metrics = {metric: sum(values) / len(values) for metric, values in fold_metrics.items()}
        print("\n=== k-Fold Validation Results ===")
        for metric, avg_value in avg_metrics.items():
            print(f"{metric.name}: {avg_value:.2f}")

        return avg_metrics