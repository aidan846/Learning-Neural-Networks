# Utilized chatgpt to make and explain the next steps from p8 using concepts from P9 which no code happened
# After P9, he stopped making videos, so im on my own

# Introducing Optimization
# First optimization experiment: modify the parameters, calculate the new loss, and keep changes that improve it

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # clip infinite values

        # Scalar class values:
        # [0, 1, 1]
        if len(y_true.shape) == 1: # if true scalar class values
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # One-hot encoded:
        # [[1, 0, 0],
        #  [0, 1, 0],
        #  [0, 1, 0]]
        elif len(y_true.shape) == 2: # one hot encoded vectors
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
            # np.sum( [0.7, 0.1, 0.2],     *   [[1, 0, 0],             = [0.7,
            #         [0.1, 0.5, 0.4],          [0, 1, 0],                0.5,
            #         [0.02, 0.9, 0.08]]        [0, 1, 0]], axis=1)       0.9]

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Generate training data
X, y = spiral_data(samples=100, classes=3) # Input is X,Y, so input is 3.

# First layer:
# 2 inputs -> 3 neurons
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Second layer:
# 3 inputs -> 3 output classes
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()


# Initial forward pass
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y) 

print("Initial Loss:", loss)


# Optimization
# We don't know HOW each weight affects the loss yet
#
# So for now, we randomly change the weights and biases.
# If the new parameters give us a slower loss, we keep them.
# Otherwise, we go back to the old parameters

lowest_loss = loss

# Save the best parameters we've found so far
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()

best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Make small random changes to every weight and bias
    dense1.weights += 0.05 * np.random.randn(*dense1.weights.shape)
    dense1.biases += 0.05 * np.random.randn(*dense1.biases.shape)

    dense2.weights += 0.05 * np.random.randn(*dense2.weights.shape)
    dense2.biases += 0.05 * np.random.randn(*dense2.biases.shape)

    # Perform a forward pass with the new parameters.
    # We do this to basically re-run the same network as before, but new parameters.
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate the new loss with the new parameters
    loss = loss_function.calculate(activation2.output, y)


    # If the loss got LOWER, keep them
    # If the loss got HIGHER, restore previous best
    if loss < lowest_loss:
        print("New best loss:", loss, "iteration:", iteration)

        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

print("\nFinal Loss:", lowest_loss)



'''
Overview of changes
---------------------------------------
In P8, the neural network does:

    X
    ↓
    Dense Layer
    ↓
    ReLU
    ↓
    Dense Layer
    ↓
    Softmax
    ↓
    Loss

Then it stops.

We might get something like "Loss: 1.0986", but the network has no idea
what to do with that information.

P9 introduces optimization, which is the process of adjusting the network's
parameters (weights and biases) in order to minimize the loss.

                ┌─────────────────┐
                │ Change weights  │
                │ and biases      │
                └────────┬────────┘
                         ↓
X → Dense → ReLU → Dense → Softmax → Loss
                         ↑                │
                         │                │
                         └────────────────┘
                           Did loss improve?


For now, we make small random changes to the weights and biases and keep
those changes if they lower the loss.

However, randomly changing parameters is not a practical way to train a
neural network. As the number of parameters grows, randomly searching for
better values becomes extremely inefficient.

For example:

    dense1 = Layer_Dense(2, 3)

Dense1 has:
    6 weights
    3 biases

Dense2 has:
    9 weights
    3 biases

So even this tiny network has 21 trainable parameters.

Real neural networks can have millions or even billions of parameters, so
randomly doing:

    weight += random_number

and then asking:

    "Did the loss improve?"

would be extremely inefficient.

The next step is to introduce derivatives and gradients. These allow us to
determine how changes to each weight and bias affect the loss, so we can
adjust the parameters in a more informed direction instead of guessing.




After Running P9.
The optimizer successfully lowers the loss, but very slowly.

This happens because we are randomly changing all of the weights and biases
without knowing which changes will help.

The next goal is to calculate how each parameter affects the loss.

That is where derivatives and gradients come in.
'''

