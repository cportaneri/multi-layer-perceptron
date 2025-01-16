import math
import random
from abc import ABC, abstractmethod

class WeightsInit(ABC):
    @abstractmethod
    def init(self, layer):
        pass

class HeNormal(WeightsInit):
    def init(self, layer):
        stddev = math.sqrt(2 / layer.previous_layer_size)
        return [random.gauss(0, stddev) for _ in range(layer.previous_layer_size)]

class HeUniform(WeightsInit):    
    def init(self, layer):
        limit = math.sqrt(6 / layer.previous_layer_size)
        return [random.uniform(-limit, limit) for _ in range(layer.previous_layer_size)]

class XavierNormal(WeightsInit):
    def init(self, layer):
        stddev = math.sqrt(2 / (layer.previous_layer_size + layer.layer_size))
        return [random.gauss(0, stddev) for _ in range(layer.previous_layer_size)]

class XavierUniform(WeightsInit):
    def init(self, layer):
        limit = math.sqrt(6 / (layer.previous_layer_size + layer.layer_size))
        return [random.uniform(-limit, limit) for _ in range(layer.previous_layer_size)]

class LecunNormal(WeightsInit):
    def init(self, layer):
        stddev = math.sqrt(1 / layer.previous_layer_size)
        return [random.gauss(0, stddev) for _ in range(layer.previous_layer_size)]