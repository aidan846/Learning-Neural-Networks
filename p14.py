import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


'''
Overview
---------------------------------------
In P13, we completed backpropagation.

After the backward pass, each dense layer had:
    dweights
    dbiases

These gradients tell us how each parameter affects the loss.

Now we can finally use that information to update the parameters.

The simplest optimizer is:
    Stochastic Gradient Descent (SGD)

The update rule is:
    parameter -= learning_rate * gradient

Why subtract?

Because the gradient points in the direction of increasing loss.

So if we want to reduce the loss, we move in the opposite direction.

Training loop:
    forward pass
        ↓
    calculate loss
        ↓
    backward pass
        ↓
    calculate gradients
        ↓
    update weights and biases
        ↓
    repeat
'''

# See P13 for the code with comments
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

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = (np.diagflat(single_output) - np.dot(single_output, single_output.T))

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # clip infinite values

        # Scalar class values
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # One-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        labels = len(dvalues[0])

        # Convert scalar to one-hot form.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = (-y_true / dvalues) / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        '''
        Gradient points uphill.

        We want to go downhill.
        So: weight = weight - learning_rate * gradient
        '''
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# =========================================================
# Generate Data
# =========================================================
X, y = spiral_data(samples=100, classes=3) # Input is X,Y, so input is 3.

# First layer:
# 2 inputs -> 3 neurons
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Second layer:
# 3 inputs (3 outputs from previous) -> 3 output classes
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=1.0)


# =========================================================
# Training Loop
# =========================================================
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate Loss
    loss = loss_function.calculate(activation2.output, y) 

    # Calculate Accuracy
    '''
    Softmax output might look like: [0.2, 0.7, 0.1]
    np.argmax gives us: 1
    which is the class with the highest probability.
    '''

    predictions = np.argmax(activation2.output, axis=1)

    accuracy = np.mean(predictions == y)

    # Print every 100 epochs
    if not epoch % 100:
        print("epoch:", epoch, "loss:", loss, "acc:", accuracy)


    # Backward Pass
    loss_function.backward(activation2.output, y)

    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update Parameters
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

'''
Summary
---------------------------------------

We now have a complete training loop.

Each epoch does:

    1. Forward pass
    2. Calculate loss
    3. Calculate accuracy
    4. Backward pass
    5. Calculate gradients
    6. Update weights and biases


The key step is:

    parameter -= learning_rate * gradient


The gradient tells us:

    "Which direction increases the loss?"


So we subtract it:

    weights -= learning_rate * dweights

and move in the opposite direction.


The learning rate controls the size of each update.

Small learning rate:

    slow, careful updates

Large learning rate:

    faster, but can overshoot


Unlike P9, we are no longer randomly guessing parameter changes.

P9:

    random change
    ↓
    check loss
    ↓
    maybe keep it


P14:

    calculate gradient
    ↓
    know which direction changes loss
    ↓
    deliberately update parameter


Next:

    P15 - Learning rate decay and momentum

These help SGD train more smoothly and efficiently.
'''