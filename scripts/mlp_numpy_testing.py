from mlp.enums import MLPType, MetricType
from scripts.utils.testing import test_model

if __name__ == "__main__":
    config = {
        "model_type": MLPType.NUMPY,
        "model_path": "models/mlp_numpy.json",
        "pixel_max_value": 255.0,
        "classes_number": 10,
        "metric": [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1]
    }
    test_model(config)