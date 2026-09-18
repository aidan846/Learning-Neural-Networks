import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

'''
Overview
---------------------------------------
In P14, we introduced SGD:

    parameter -= learning_rate * gradient

This works, but plain SGD has two problems:

1. The learning rate stays the same forever.
2. Updates can bounce back and forth instead of moving smoothly.

P15 introduces:

    Learning rate decay
    Momentum

Learning rate decay gradually lowers the learning rate as training continues.

Momentum remembers part of the previous update and uses it to help guide
the next one.
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

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
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


# Majority of differences from P14 here.
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        # Gradually lower the learning rate as training continues.
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            # Create momentum arrays the first time this layer is updated.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Previous momentum is kept, then current gradient is added.
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


# Generate training data
X, y = spiral_data(samples=100, classes=3)

# Network
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# SGD with learning rate decay and momentum
optimizer = Optimizer_SGD(
    learning_rate=1.0,
    decay=1e-3,
    momentum=0.9
)


for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Loss
    loss = loss_function.calculate(activation2.output, y)

    # Accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            "epoch:", epoch,
            "loss:", loss,
            "acc:", accuracy,
            "lr:", optimizer.current_learning_rate
        )

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)

    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update parameters

    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()


'''
Summary
---------------------------------------

P14 used plain SGD:

    weight -= learning_rate * gradient


P15 adds two improvements:

    1. Learning rate decay
    2. Momentum


LEARNING RATE DECAY
---------------------------------------

At the start:

    learning rate = 1.0

As training continues:

    learning rate gets smaller

Formula:

    current_lr = starting_lr / (1 + decay * iteration)

This lets the network take larger steps early in training and smaller,
more precise steps later.

MOMENTUM
---------------------------------------

Plain SGD only looks at the current gradient.

Momentum also remembers part of the previous update:

    new_update =
        momentum * previous_update
        -
        learning_rate * gradient

For example:

    momentum = 0.9

means we keep 90% of the previous movement.

This can help the optimizer:

    move faster in consistent directions
    reduce bouncing
    move through shallow areas more smoothly

The optimizer now has three stages:

    optimizer.pre_update_params()

        ↓

    optimizer.update_params(layer)

        ↓

    optimizer.post_update_params()

pre_update_params():
    Calculates the current learning rate.

update_params():
    Updates weights and biases.

post_update_params():
    Increases the iteration counter.


Next:
    P16 - AdaGrad, RMSprop, and Adam
'''