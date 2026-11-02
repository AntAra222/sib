
import numpy as np
from abc import ABCMeta, abstractmethod


# Classe base Layer
class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__



# Classe DenseLayer
class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None):

        n_features = self._input_shape[0]

        # Pesos aleatorios entre -0.5 e 0.5
        self.weights = (
            np.random.rand(n_features, self.n_units) - 0.5
        )

        # Bias a zero
        self.biases = np.zeros((1, self.n_units))

        # Inicializar os otimizadores
        if optimizer is not None:
            from copy import deepcopy
            self.w_opt = deepcopy(optimizer)
            self.b_opt = deepcopy(optimizer)

        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):

        self.input = input

        self.output = (
            np.dot(input, self.weights) + self.biases
        )

        return self.output

    def backward_propagation(self, output_error):

        # 1. Calcular input_error (antes de atualizar pesos)
        input_error = np.dot(output_error, self.weights.T)

        # 2. Calcular weights_error
        weights_error = np.dot(self.input.T, output_error)

        # 3. Calcular bias_error
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # 4. Atualizar pesos e bias
        self.weights = self.w_opt.update(
            self.weights, weights_error
        )

        self.biases = self.b_opt.update(
            self.biases, bias_error
        )

        # 5. Devolver input_error
        return input_error

    def output_shape(self):
        return (self.n_units,)



# Classe ActivationLayer
class ActivationLayer(Layer):

    def __init__(self, input_shape=None):
        self._input_shape = input_shape
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=False):

        # Guardar a entrada
        self.input = input

        # Aplicar a funcao de ativacao
        self.output = self.activation_function(input)

        # Devolver a saida
        return self.output

    def backward_propagation(self, output_error):

        # Calcular o erro da entrada
        input_error = self.derivative(self.input) * output_error

        # Devolver o erro
        return input_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0



# Classe Dropout
class Dropout(Layer):

    def __init__(self, probability):
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None
        self._input_shape = None

    def forward_propagation(self, input, training=False):

        self.input = input

        if training:

            # Calcular o fator de escala
            scale = 1 / (1 - self.probability)

            # Criar a mascara binomial
            self.mask = np.random.binomial(
                1, 1 - self.probability, size=input.shape
            )

            # Calcular a saida
            self.output = input * self.mask * scale

        else:

            # Em inference nao altera a entrada
            self.mask = None
            self.output = input

        return self.output

    def backward_propagation(self, output_error):

        # Multiplicar o erro pela mascara
        if self.mask is None:
            return output_error

        scale = 1 / (1 - self.probability)
        input_error = output_error * self.mask * scale

        return input_error

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0
