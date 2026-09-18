import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from plot import SpiralVisualizer

nnfs.init()
'''
Overview
---------------------------------------
In P17, our network became much faster by combining:

    Softmax
    +
    Categorical Cross Entropy

Now the network can fit the training data very well.

But this introduces another problem:

    overfitting

Overfitting happens when the network learns the training data too specifically.

It may get very high training accuracy, but perform worse on new data.

Regularization helps reduce this by adding a penalty for large weights
and biases.

We will introduce:

    L1 regularization
    L2 regularization
'''

class Layer_Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0
    ):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # NEW: Store regularization strengths for this layer
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # NEW: L1 regularization gradients
        '''
        L1 penalty uses:

            abs(weight)

        Its derivative is roughly:

            +1 if weight > 0
            -1 if weight < 0

        This pushes weights toward zero.
        '''
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1

            self.dweights += self.weight_regularizer_l1 * dL1

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1

            self.dbiases += self.bias_regularizer_l1 * dL1

        # NEW: L2 regularization gradients

        '''
        L2 penalty uses:

            weight^2

        Derivative:

            2 * weight

        So large weights receive a larger penalty.
        '''

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

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

class Loss:
    def regularization_loss(self, layer):
        regularization_loss = 0

        # L1 weight penalty
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L1 bias penalty
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2 weight penalty
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L2 bias penalty
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

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


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
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


# Data
X, y = spiral_data(samples=100, classes=3)

# Network
# NEW: L2 regularization is usually the most common choice.
#
# Here we apply it to the weights of the first layer.

dense1 = Layer_Dense(
    2,
    100,
    weight_regularizer_l2=5e-4
)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(
    100,
    3,
    weight_regularizer_l2=5e-4
)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(
    learning_rate=0.05,
    decay=5e-7
)

visualizer = SpiralVisualizer(X, y)
def model_predict(inputs):
    dense1.forward(inputs)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.activation.forward(dense2.output)
    return np.argmax(loss_activation.activation.output, axis=1)

# Training
for epoch in range(10001):

    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    # -----------------------------------------------------
    # NEW: Data loss + regularization loss
    # -----------------------------------------------------

    '''
    Our total loss is now:

        total loss =
            data loss
            +
            regularization loss

    Data loss tells us how wrong the predictions are.

    Regularization loss penalizes large parameters.
    '''

    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Accuracy 
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            "epoch:", epoch,
            "acc:", accuracy,
            "loss:", loss,
            "data_loss:", data_loss,
            "reg_loss:", regularization_loss,
            "lr:", optimizer.current_learning_rate
        )

        if not epoch % 500:
            visualizer.update(epoch, model_predict)

            dense1.forward(X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss_activation.forward(dense2.output, y)

    # Backward
    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()

visualizer.show_final(model_predict)

'''
Summary
---------------------------------------

Before P18, our loss only measured prediction error:

    loss = data loss


Now:

    total loss =
        data loss
        +
        regularization loss


REGULARIZATION
---------------------------------------

Regularization discourages the network from relying on very large
weights and biases.

This can help reduce overfitting.


L1 REGULARIZATION
---------------------------------------

L1 adds:

    abs(weight)

to the loss.

Example:

    weight = -4

L1 penalty uses:

    abs(-4) = 4


Its gradient is roughly:

    +1 for positive weights
    -1 for negative weights


This pushes weights directly toward zero.

L1 can cause some weights to become very small or exactly zero.

This can make the network more sparse.


L2 REGULARIZATION
---------------------------------------

L2 adds:

    weight^2

to the loss.

Example:

    weight = 4

L2 penalty:

    4^2 = 16


Its derivative is:

    2 * weight


This means large weights receive a much stronger penalty than small ones.

L2 usually does not force weights exactly to zero.

Instead, it encourages the network to spread information across many
smaller weights.


Example:

    one weight = 10

gets penalized much more than:

    ten weights = 1


WHY ADD REGULARIZATION TO THE GRADIENT?
---------------------------------------

It is not enough to add regularization to the displayed loss.

The optimizer needs to know about it too.

So during backward():

    original gradient
        +
    regularization gradient

becomes:

    final gradient


For L2:

    dweights += 2 * regularization_strength * weights


This means Adam now updates parameters based on:

    prediction error
        +
    regularization penalty


COMMON USAGE
---------------------------------------

L2 is generally the more common choice.

Example:

    Layer_Dense(
        2,
        64,
        weight_regularizer_l2=5e-4
    )


The regularization strength controls how strongly we penalize large weights.

Too small:

    almost no effect

Too large:

    network becomes too restricted and may underfit


The goal is not:

    make every weight tiny

The goal is:

    prevent the network from depending on unnecessarily extreme parameters


Next:
    P19 - Dropout

Dropout temporarily disables random neurons during training.

This forces the network to avoid depending too heavily on specific
neurons and provides another way to reduce overfitting.
'''