from enum import Enum

class MLPType(Enum):
    BASE = "base"
    NUMPY = "numpy"

class ActivationType(Enum):
    RELU = "relu"
    SOFTMAX ="softmax"
    SIGMOID = "sigmoid"
    TANH = "tanh"

class WeightsInitType(Enum):
    HNORM = "he normal"
    HUNI = "he uniform"
    XNORM = "xavier normal"
    XUNI = "xavier uniform"
    LNORM = "lecun normal"
    
class OptimizerType(Enum):
    SGD = "stochastic gradient descent"
    ADAM = "adam"

class MetricType(Enum):
    MSE = "mean squared error"
    CROSS = "cross entropy"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1-score"

class ValidationType(Enum):
    HOLDOUT = "hold-out"
    KFOLD = "k-fold"

class HyperparametersSearchType(Enum):
    GRID = "grid"
    RANDOM = "random"