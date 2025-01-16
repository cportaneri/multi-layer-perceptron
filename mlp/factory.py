from mlp.enums import MLPType, ActivationType, WeightsInitType, OptimizerType, MetricType, ValidationType, HyperparametersSearchType
from mlp.metric import Accuracy, Precision, Recall, F1Score
from mlp.validation import HoldOut, KFold
from mlp.hyperparameter import GridSearch, RandomSearch
from mlp.base.mlp import MultiLayerPerceptron 
from mlp.base.optimizer import StochasticGradientDescent, Adam
from mlp.base.activation import Relu, Softmax, Sigmoid, Tanh
from mlp.base.weights_init import HeNormal, HeUniform, XavierNormal, XavierUniform, LecunNormal
from mlp.numpy.weights_init import HeNormalNumpy, XavierNormalNumpy
from mlp.numpy.mlp import MultiLayerPerceptronNumpy
from mlp.numpy.optimizer import StochasticGradientDescentNumpy, AdamNumpy
from mlp.numpy.activation import ReluNumpy, SoftmaxNumpy, SigmoidNumpy, TanhNumpy

class Factory:
    @staticmethod
    def create_mlp(mlp_type: MLPType):
        if mlp_type == MLPType.BASE:
            return MultiLayerPerceptron()
        elif mlp_type == MLPType.NUMPY:
            return MultiLayerPerceptronNumpy()
        else:
            raise ValueError("Invalid MLP type specified")

    @staticmethod
    def create_activation(mlp_type: MLPType, activation_type: ActivationType):
        if mlp_type == MLPType.BASE:
            if activation_type == ActivationType.RELU:
                return Relu()
            elif activation_type == ActivationType.SOFTMAX:
                return Softmax()
            elif activation_type == ActivationType.SIGMOID:
                return Sigmoid()
            elif activation_type == ActivationType.TANH:
                return Tanh()
            else:
                raise ValueError("Invalid activation type specified")
        elif mlp_type == MLPType.NUMPY:
            if activation_type == ActivationType.RELU:
                return ReluNumpy()
            elif activation_type == ActivationType.SOFTMAX:
                return SoftmaxNumpy()
            elif activation_type == ActivationType.SIGMOID:
                return SigmoidNumpy()
            elif activation_type == ActivationType.TANH:
                return TanhNumpy()
            else:
                raise ValueError("Invalid activation type specified")
        else:
            raise ValueError("Invalid mlp type specified")

    @staticmethod
    def create_weights_init(mlp_type: MLPType, weights_init_type: WeightsInitType):    
        if mlp_type == MLPType.BASE:
            if weights_init_type == WeightsInitType.HNORM:
                return HeNormal()
            elif weights_init_type == WeightsInitType.HUNI:
                return HeUniform()
            elif weights_init_type == WeightsInitType.XNORM:
                return XavierNormal()
            elif weights_init_type == WeightsInitType.XUNI:
                return XavierUniform()
            elif weights_init_type == WeightsInitType.LNORM:
                return LecunNormal()
            else:
                raise ValueError("Invalid Weights init type specified")
        elif mlp_type == MLPType.NUMPY:
            if weights_init_type == WeightsInitType.HNORM:
                return HeNormalNumpy()
            elif weights_init_type == WeightsInitType.XNORM:
                return XavierNormalNumpy()
            else:
                raise ValueError("Invalid Weights init specified")
        else:
            raise ValueError("Invalid mlp type specified")
        
    @staticmethod
    def create_optimizer(mlp_type: MLPType, optimizer_type: OptimizerType):
        if mlp_type == MLPType.BASE:
            if optimizer_type == OptimizerType.SGD:
                return StochasticGradientDescent()
            elif optimizer_type == OptimizerType.ADAM:
                return Adam()
            else:
                raise ValueError("Invalid optimizer type specified")
        elif mlp_type == MLPType.NUMPY:
            if optimizer_type == OptimizerType.SGD:
                return StochasticGradientDescentNumpy()
            elif optimizer_type == OptimizerType.ADAM:
                return AdamNumpy()
            else:
                raise ValueError("Invalid optimizer type specified")
        else:
            raise ValueError("Invalid mlp type specified")

    @staticmethod 
    def create_metric(mlp_type: MLPType, metric_type: MetricType):
        if metric_type == MetricType.ACCURACY:
            return Accuracy()
        elif metric_type == MetricType.PRECISION:
            return Precision()
        elif metric_type == MetricType.RECALL:
            return Recall()
        elif metric_type == MetricType.F1:
            return F1Score()
        elif metric_type == MetricType.MSE:
            from mlp.base.loss import MeanSquaredError
            from mlp.numpy.loss import MeanSquaredErrorNumpy
            if mlp_type == MLPType.BASE:
                return MeanSquaredError()
            elif mlp_type == MLPType.NUMPY:
                return MeanSquaredErrorNumpy()
            else:
                raise ValueError("Invalid activation type specified")
        elif metric_type == MetricType.CROSS:
            from mlp.base.loss import CrossEntropy
            from mlp.numpy.loss import CrossEntropyNumpy
            if mlp_type == MLPType.BASE:
                return CrossEntropy()
            elif mlp_type == MLPType.NUMPY:
                return CrossEntropyNumpy()
            else:
                raise ValueError("Invalid activation type specified")
        else:
            raise ValueError("Invalid mlp type specified")

    @staticmethod  
    def create_validation(validation_type: ValidationType):
        if validation_type == ValidationType.HOLDOUT:
            return HoldOut()
        elif validation_type == ValidationType.KFOLD:
            return KFold()
        else:
            raise ValueError("Invalid validation type specified")

    @staticmethod
    def create_hyperparameters_search(hyperparameters_search_type: HyperparametersSearchType):
        if hyperparameters_search_type == HyperparametersSearchType.GRID:
            return GridSearch()
        elif hyperparameters_search_type == HyperparametersSearchType.RANDOM:
            return RandomSearch()
        else:
            raise ValueError("Invalid hyperparameters search type specified")