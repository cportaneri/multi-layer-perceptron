import numpy as np
from mlp.metric import LossMetric
from mlp.enums import MetricType

class MeanSquaredErrorNumpy(LossMetric):
    def type(self):
        return MetricType.MSE
    
    def compute_average(self, predictions, targets):
        super().compute_average(predictions, targets)

    def compute(self, prediction, target):
        return np.mean((prediction - target) ** 2)

class CrossEntropyNumpy(LossMetric):
    def type(self):
        return MetricType.CROSS
    
    def compute_average(self, predictions, targets):
        super().compute_average(predictions, targets)
        
    def compute(self, prediction, target):
        prediction = np.clip(prediction, 1e-8, 1.0)
        return -np.sum(target * np.log(prediction)) / target.shape[1]