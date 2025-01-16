from datasets import load_dataset
from scripts.utils.preprocessing import preprocess_input, preprocess_output
from mlp.io import save_neural_network

def train_model(config):
    print("Loading MNIST datasets...")
    training_dataset = load_dataset("ylecun/mnist", split="train")
    testing_dataset = load_dataset("ylecun/mnist", split="test")

    print("Preprocessing dataset...")
    training_dataset_inputs = preprocess_input(training_dataset, config["pixel_max_value"])
    training_dataset_outputs = preprocess_output(training_dataset, config["classes_number"])
    testing_dataset_inputs = preprocess_input(testing_dataset, config["pixel_max_value"])
    testing_dataset_outputs = preprocess_output(testing_dataset, config["classes_number"])

    print("Starting hold-out validation...")
    from mlp.neural_network import NeuralNetwork
    nn = NeuralNetwork(config["model_type"])

    for layer in config["layers"]:
        nn.add_layer(
            layer_dim=layer["layer_dim"],
            activation_type=layer["activation"],
            dropout_rate=layer.get("dropout_rate", 0.0),
            input_dim=layer.get("input_dim", -1)
        )

    results = nn.validation(
        validation_type=config["validation_type"],
        learning_rate=config["learning_rate"],
        max_epoch=config["max_epoch"],
        batch_size=config["batch_size"],
        l2_lambda=config["l2_lambda"],
        loss_function_type=config["loss_function"],
        optimizer_type=config["optimizer"],
        metrics=config["metric"],
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
    save_neural_network(nn, config["model_path"])