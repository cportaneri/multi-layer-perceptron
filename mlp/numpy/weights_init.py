import numpy as np
from abc import ABC, abstractmethod

class WeightsInitNumpy(ABC):
    @abstractmethod
    def init(self, layer):
        pass

class HeNormalNumpy(WeightsInitNumpy):
    def init(self, layer):
        stddev = np.sqrt(2 / layer.previous_layer_size)
        return np.random.normal(0, stddev, (layer.layer_size, layer.previous_layer_size))
              
class XavierNormalNumpy(WeightsInitNumpy):
    def init(self, layer):
        stddev = np.sqrt(2 / (layer.previous_layer_size + layer.layer_size))
        return np.random.normal(0, stddev, (layer.layer_size, layer.previous_layer_size))