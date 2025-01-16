from mlp.enums import MLPType, OptimizerType, MetricType, ValidationType, ActivationType
from scripts.utils.training import train_model

if __name__ == "__main__":
    config = {
        "model_type": MLPType.BASE,
        "model_path": "models/mlp.json",
        "pixel_max_value": 255.0,
        "classes_number": 10,
        "learning_rate": 0.0002,
        "max_epoch": 15,
        "batch_size": 32,
        "l2_lambda": 0.0001,
        "optimizer": OptimizerType.ADAM,
        "loss_function": MetricType.CROSS,
        "metric": [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1],
        "validation_type": ValidationType.HOLDOUT,
        "layers": [
            {"layer_dim": 45, "activation": ActivationType.RELU, "dropout_rate": 0.3, "input_dim": 28*28},
            {"layer_dim": 35, "activation": ActivationType.RELU},
            {"layer_dim": 23, "activation": ActivationType.RELU},
            {"layer_dim": 10, "activation": ActivationType.SOFTMAX}
        ]
    }
    train_model(config)