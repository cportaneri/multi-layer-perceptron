
# MNIST Multi-Layer Perceptron Classifier

This repository contains a custom implementation of a Multi-Layer Perceptron (MLP) for classifying MNIST handwritten digits made for educational purposes. The project includes two main implementations:

- **Base version**: A simple implementation using only Python and its math library. This version is less performant and efficient computationally but provides a clearer understanding of the algorithm, making it ideal for learning how MLPs work from scratch.
- **Numpy version**: A more optimized version that leverages NumPy for matrix operations, resulting in better computational efficiency and performance.

The repository also provides scripts to compare these custom implementations with TensorFlow/Keras and PyTorch.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/cportaneri/multi-layer-perceptron.git
   cd multi-layer-perceptron
   ```
2. Install the necessary dependencies (you can uncomment the lines in requirements.txt for optional libraries):
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The API for training and testing the neural network is provided by the following exemple:

### Training API:
```python
from datasets import load_dataset
from mlp.enums import MLPType, OptimizerType, MetricType, ValidationType, ActivationType
from scripts.utils.preprocessing import preprocess_input, preprocess_output
from mlp.io import save_neural_network

print("Loading MNIST datasets...")
training_dataset = load_dataset("ylecun/mnist", split="train")
testing_dataset = load_dataset("ylecun/mnist", split="test")

print("Preprocessing dataset...")
training_dataset_inputs = preprocess_input(training_dataset, 255.0)
training_dataset_outputs = preprocess_output(training_dataset, 10)
testing_dataset_inputs = preprocess_input(testing_dataset, 255.0)
testing_dataset_outputs = preprocess_output(testing_dataset, 10)

print("Starting hold-out validation...")
from mlp.neural_network import NeuralNetwork
nn = NeuralNetwork(MLPType.NUMPY)
nn.add_layer(layer_dim=45, activation_type=ActivationType.RELU, dropout_rate=0.3, input_dim=28*28),
nn.add_layer(layer_dim=35, activation_type=ActivationType.RELU),
nn.add_layer(layer_dim=23, activation_type=ActivationType.RELU),
nn.add_layer(layer_dim=10, activation_type=ActivationType.SOFTMAX)

results = nn.validation(
   validation_type=ValidationType.HOLDOUT,
   learning_rate=0.0002,
   max_epoch=30,
   batch_size=32,
   l2_lambda=0.0001,
   loss_function_type=MetricType.CROSS,
   optimizer_type=OptimizerType.ADAM,
   metrics=[MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1],
   training_dataset_inputs=training_dataset_inputs,
   training_dataset_outputs=training_dataset_outputs,
   testing_dataset_inputs=testing_dataset_inputs,
   testing_dataset_outputs=testing_dataset_outputs
)

print("Results : ")
print(f" Accuracy: {results[0]:.2f}%")
print(f" Precision: {results[1]:.2f}%")
print(f" Recall: {results[2]:.2f}%")
print(f" F1-Score: {results[3]:.2f}%")
nn.print_configuration()

print("Saving model...")
save_neural_network(nn, "models/mlp_numpy.json")
```

### Testing API:
```python
from datasets import load_dataset
from mlp.enums import MLPType, MetricType
from scripts.utils.preprocessing import preprocess_input, preprocess_output
from mlp.io import load_neural_network

def test_model(config):
    print("Loading MNIST testing dataset...")
    testing_dataset = load_dataset("ylecun/mnist", split="test")

    print("Preprocessing testing dataset...")
    testing_dataset_inputs = preprocess_input(testing_dataset, 255.0)
    testing_dataset_outputs = preprocess_output(testing_dataset, 10)

    print("Loading neural network...")
    from mlp.neural_network import NeuralNetwork
    nn = NeuralNetwork(MLPType.NUMPY)
    load_neural_network(nn, "models/mlp_numpy.json")

    print("Testing neural network...")
    results = nn.test(testing_dataset_inputs, testing_dataset_outputs, [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1])

    print("Results : ")
    print(f" Accuracy: {results[0]:.2f}%")
    print(f" Precision: {results[1]:.2f}%")
    print(f" Recall: {results[2]:.2f}%")
    print(f" F1-Score: {results[3]:.2f}%")
    nn.print_configuration()
```

### Configuration
The configuration possibilities for training and testing is defined in the mlp/enums.py file. Key parameters include:
- `MLPType`: Choose between base or numpy.
- `ActivationType`: Select the activation type (ReLU, Sigmoid, Tanh and Softmax).
- `WeightsInitType`: Choose how the weights are initialised (He, Xavier or Lecun; Normal or Uniform).
- `OptimizerType`: Select the optimizer (SGD, Adam).
- `MetricType`: Select the loss function (MSE, cross-entropy) and classification metrics (accuracy, precision, recall, or F1-score)
- `ValidationType`: Choose the validation algorithm between standard hold-out and k-fold.
- `HyperparametersSearchType`: Choose the tuning algorithm between grid or random search

## Benchmarking Scripts
The project includes several scripts that train or test a basic 4 layers model with predefined parameters for benchmarking purposes.
You can run these scripts from the command line like so:

#### Training Example (Base version):
```bash
python -m scripts.mlp_base_training
```
#### Testing Example (Base version):
```bash
python -m scripts.mlp_base_testing
```
#### Training Example (NumPy version):
```bash
python -m scripts.mlp_numpy_training
```
#### Testing Example (NumPy version):
```bash
python -m scripts.mlp_numpy_testing
```

To compare the model with implementations in standard neural network libraries like PyTorch or Keras, you can run the following scripts:
#### Training Example (PyTorch):
```bash
python scripts/framework/mlp_pytorch_training.py
```
#### Training Example (Keras):
```bash
python scripts/framework/mlp_keras_training.py
```

### Results
The training is conducted on the MNIST test set using a simple model with 4 layers as described in the scripts above.
This implementation only use CPU. I use a i9-13900KF.
- The base implementation achieves ~82% accuracy for a ~30 min training. This low result can be attributed to the numerical instability inherent in Python's basic math operations.
- The NumPy implementation achieves ~93% accuracy in 85 seconds.
- PyTorch implementations achieve ~95.5% accuracy in 205 seconds.
- Keras implementations achieve ~97% accuracy in 45 seconds.

## License
This project is licensed under the MIT License.
