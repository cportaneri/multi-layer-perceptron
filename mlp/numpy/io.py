import numpy as np
from mlp.numpy.mlp import MultiLayerPerceptronNumpy
from mlp.numpy.layer import SingleLayerPerceptronNumpy
from mlp.enums import ActivationType

def save_mlp_numpy(mlp : MultiLayerPerceptronNumpy):
    mlp_data = []
    for layer in mlp.layers:
        mlp_data.append({
            "layer_size": layer.layer_size,
            "previous_layer_size": layer.previous_layer_size,
            "activation_type": layer.activation.type().name,
            "dropout_rate": layer.dropout_rate,
            "weights": layer.weights.tolist(),
            "bias": layer.bias.tolist(),
        })

    return mlp_data

def load_mlp_numpy(mlp : MultiLayerPerceptronNumpy, mlp_data):
    from mlp.factory import Factory
    mlp.layers = []
    for layer_data in mlp_data:
        layer = SingleLayerPerceptronNumpy(
            layer_data["layer_size"],
            layer_data["previous_layer_size"],
            Factory.create_activation(mlp.type(), ActivationType[layer_data["activation_type"]]),
            layer_data["dropout_rate"]
        )
        layer.weights = np.array(layer_data["weights"])
        layer.bias = np.array(layer_data["bias"])
        mlp.layers.append(layer)