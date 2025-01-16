from mlp.base.mlp import MultiLayerPerceptron
from mlp.base.layer import SingleLayerPerceptron
from mlp.enums import ActivationType

def save_mlp(mlp : MultiLayerPerceptron):
    mlp_data = []
    for layer in mlp.layers:
        mlp_data.append({
            "layer_size": layer.layer_size,
            "previous_layer_size": layer.previous_layer_size,
            "activation_type": layer.activation.type().name,
            "dropout_rate": layer.dropout_rate,
            "neurons": [{
                "weights": neuron.weights,
                "bias": neuron.bias
            } for neuron in layer.neurons]
        })

    return mlp_data

def load_mlp(mlp : MultiLayerPerceptron, mlp_data):
    from mlp.factory import Factory
    mlp.layers = []
    for layer_data in mlp_data:
        layer = SingleLayerPerceptron(
            layer_data["layer_size"],
            layer_data["previous_layer_size"],
            Factory.create_activation(mlp.type(), ActivationType[layer_data["activation_type"]]),
            layer_data["dropout_rate"]
        )
        for neuron, neuron_data in zip(layer.neurons, layer_data["neurons"]):
            neuron.weights = neuron_data["weights"]
            neuron.bias = neuron_data["bias"]
        mlp.layers.append(layer)