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

    
# Classe ReLUActivation
class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return (input > 0).astype(float)


# Classe SigmoidActivation
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)


# Classe TanhActivation
class TanhActivation(ActivationLayer):

    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        return 1 - np.tanh(input) ** 2


# Classe SoftmaxActivation
class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):

        # Evitar valores exponenciais muito grandes
        exp_values = np.exp(
            input - np.max(input, axis=-1, keepdims=True)
        )

        # Calcular softmax
        return exp_values / np.sum(
            exp_values, axis=-1, keepdims=True
        )

    def derivative(self, input):

        s = self.activation_function(input)

        return s * (1 - s)
