
from datasets import load_dataset
from scripts.utils.preprocessing import preprocess_input, preprocess_output
from mlp.io import load_neural_network

def test_model(config):
    print("Loading MNIST testing dataset...")
    testing_dataset = load_dataset("ylecun/mnist", split="test")

    print("Preprocessing testing dataset...")
    testing_dataset_inputs = preprocess_input(testing_dataset, config["pixel_max_value"])
    testing_dataset_outputs = preprocess_output(testing_dataset, config["classes_number"])

    print("Loading neural network...")
    from mlp.neural_network import NeuralNetwork
    nn = NeuralNetwork(config["model_type"])
    load_neural_network(nn, config["model_path"])

    print("Testing neural network...")
    results = nn.test(testing_dataset_inputs, testing_dataset_outputs, config["metric"])

    print("Results : ")
    print(f" Accuracy: {results[0]:.2f}%")
    print(f" Precision: {results[1]:.2f}%")
    print(f" Recall: {results[2]:.2f}%")
    print(f" F1-Score: {results[3]:.2f}%")
    nn.print_configuration()