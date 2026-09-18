import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

'''
Overview
---------------------------------------
In P16, Softmax and Categorical Cross Entropy were separate.

Backward pass:

    Loss.backward()
        ↓
    Softmax.backward()
        ↓
    Dense.backward()

This works, but Softmax.backward() is relatively expensive because it
builds a Jacobian matrix for every sample.

When Softmax and Categorical Cross Entropy are used together, their
derivatives simplify into a much cleaner formula.

Instead of calculating:

    Cross Entropy derivative
        ↓
    Softmax Jacobian
        ↓
    multiply them together

we can directly calculate:

    dinputs = probabilities - y_true

Then divide by the number of samples.

This gives us the same final gradient with much less work.
'''

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = (
                np.diagflat(single_output)
                - np.dot(single_output, single_output.T)
            )

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# ---------------------------------------------------------
# NEW: Combined Softmax + Cross Entropy
# ---------------------------------------------------------
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Run softmax first
        self.activation.forward(inputs)

        # Save softmax probabilities to use later
        self.output = self.activation.output

        # Calculate Cross Entropy
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        '''
        This is the important simplification.

        When Softmax and Cross Entropy are combined:

            dLoss/dInputs = predictions - y_true

        For scalar labels, like:

            [0, 2, 1]

        we can avoid converting everything into one-hot vectors.

        We copy the prediction probabilities, then subtract 1 from
        the probability of the correct class for each sample.
        '''

        samples = len(dvalues)

        # If labels are one-hot encoded, convert them to scalar class indices.
        #
        # Example:
        #
        #   [1, 0, 0] -> 0
        #   [0, 0, 1] -> 2
        #   [0, 1, 0] -> 1
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Start with the Softmax output probabilities
        self.dinputs = dvalues.copy()

        # Subtract 1 from the probabilities of the correct classes
        # Example:
        #   prediction = [0.2, 0.7, 0.1]
        #   true class = 1
        # Becomes:
        #   [0.2, -0.3, 0.1]
        # Because:
        # 0.7 - 0.1 = -0.3
        self.dinputs[range(samples), y_true] -= 1

        # Normalize the gradients accross all samples
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    def __init__(
            self,
            learning_rate=0.001,
            decay=0.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)

            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))


        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Final update weights and biases
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

X, y = spiral_data(samples=100, classes=3)

# ---------------------------------------------------------
# Network
# ---------------------------------------------------------

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)

# This replaces having separate:
#
#   activation2 = Activation_Softmax()
#   loss_function = Loss_CategoricalCrossentropy()
#
# with one combined object.

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


optimizer = Optimizer_Adam(
    learning_rate=0.05,
    decay=5e-7
)

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(10001):

    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    # NEW:
    # Softmax and Cross Entropy forward pass are now handled together.
    loss = loss_activation.forward(dense2.output, y)

    # Predictions now come from the combined object's Softmax output.
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            "epoch:", epoch,
            "loss:", loss,
            "acc:", accuracy,
            "lr:", optimizer.current_learning_rate
        )

    # -----------------------------------------------------
    # Backward
    # -----------------------------------------------------

    # NEW:
    # Instead of:
    #
    #   loss_function.backward(...)
    #   activation2.backward(...)
    #
    # we perform both operations in one simplified backward pass.
    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()


'''
Summary
---------------------------------------

Before P17, the end of the network looked like:

    Dense 2
      ↓
    Softmax
      ↓
    Cross Entropy Loss


And backward propagation looked like:

    Cross Entropy.backward()
      ↓
    Softmax.backward()
      ↓
    Dense 2.backward()


Softmax backward required a Jacobian matrix for every sample:

    jacobian =
        diag(softmax_output)
        -
        softmax_output * softmax_output.T


That works, but it is unnecessarily expensive when Softmax is immediately
followed by Categorical Cross Entropy.


When the two derivatives are combined, they simplify to:

    gradient = predictions - y_true


For scalar class labels, we can calculate this very efficiently:

    self.dinputs = predictions.copy()

    self.dinputs[
        range(samples),
        correct_classes
    ] -= 1


Then normalize:

    self.dinputs /= samples


So P17 changes:

    Loss.backward()
    Softmax.backward()

into:

    Softmax + Cross Entropy.backward()


The forward math has not changed.

We still do:

    Dense
      ↓
    Softmax
      ↓
    Cross Entropy

We are only simplifying the derivative calculation during backpropagation.


The network training flow is now:

    X
    ↓
    Dense 1
    ↓
    ReLU
    ↓
    Dense 2
    ↓
    Softmax + Cross Entropy
    ↓
    Loss


Backward:

    Softmax + Cross Entropy.backward()
    ↓
    Dense 2.backward()
    ↓
    ReLU.backward()
    ↓
    Dense 1.backward()


This is faster, simpler, and gives the same gradients.


Next:
    P18 - Regularization

We can introduce:

    L1 regularization
    L2 regularization

to discourage the network from relying on excessively large weights and
help reduce overfitting.
'''