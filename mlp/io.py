import json
from mlp.neural_network import NeuralNetwork
from mlp.factory import Factory
from mlp.enums import MLPType, OptimizerType, MetricType
from mlp.base.io import save_mlp, load_mlp
from mlp.numpy.io import save_mlp_numpy, load_mlp_numpy

def save_neural_network(nn: NeuralNetwork, filename: str):
    model_data = {
        "mlp_type": nn.mlp.type().name,
        "learning_rate": nn.learning_rate,
        "batch_size": nn.batch_size,
        "max_epoch": nn.max_epoch,
        "optimizer": nn.mlp.optimizer.type().name,
        "l2_lambda": nn.l2_lambda,
        "loss": nn.mlp.loss_function.type().name,
        "mlp": save_mlp_numpy(nn.mlp) if nn.mlp.type() == MLPType.NUMPY else save_mlp(nn.mlp)
    }

    with open(filename, 'w') as file:
        json.dump(model_data, file)
    print(f"Model saved to {filename}")

def load_neural_network(nn: NeuralNetwork, filename: str):
    with open(filename, 'r') as file:
        model_data = json.load(file)

    nn.mlp = Factory.create_mlp(MLPType[model_data["mlp_type"]])        
    nn.learning_rate = model_data["learning_rate"]
    nn.batch_size = model_data["batch_size"]
    nn.max_epoch = model_data["max_epoch"]
    nn.l2_lambda = model_data["l2_lambda"]
    nn.optimizer_type = OptimizerType[model_data["optimizer"]]
    nn.loss_function_type = MetricType[model_data["loss"]]
    nn.mlp.set_optimizer(Factory.create_optimizer(nn.mlp.type(), nn.optimizer_type))
    nn.mlp.loss_function = Factory.create_metric(nn.mlp.type(), nn.loss_function_type)
    load_mlp_numpy(nn.mlp, model_data["mlp"]) if nn.mlp.type() == MLPType.NUMPY else load_mlp(nn.mlp, model_data["mlp"])
    print(f"Model loaded from {filename}")