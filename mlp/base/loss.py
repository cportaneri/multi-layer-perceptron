import math
from mlp.metric import LossMetric
from mlp.enums import MetricType

class MeanSquaredError(LossMetric):
    def type(self):
        return MetricType.MSE
    
    def compute_average(self, predictions, targets):
        super().compute_average(predictions, targets)

    def compute(self, prediction, target):
        return sum((x[0] - x[1]) * (x[0] - x[1]) for x in zip(prediction, target)) / len(target)
    
class CrossEntropy(LossMetric):
    def type(self):
        return MetricType.CROSS
    
    def compute_average(self, predictions, targets):
        super().compute_average(predictions, targets)
        
    def compute(self, prediction, target):
        loss = 0.0
        for i in range(len(prediction)):
            p = prediction[i] + 1e-8
            t = target[i]
            if p > 0:
                loss -= t * math.log(p)
        return loss