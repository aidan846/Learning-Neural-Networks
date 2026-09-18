import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

'''
Overview
---------------------------------------
Up through P9, our neural network could perform a forward pass:
    X
    ↓
    Dense 1
    ↓
    ReLU
    ↓
    Dense 2
    ↓
    Softmax
    ↓
    Loss

In P10-P12, we learned:
    Derivatives
        ↓
    Partial derivatives
        ↓
    Chain rule

Now we can finally apply those ideas to the actual neural network.


During the forward pass, information moves:
    inputs → outputs


During backpropagation, gradients move backward:
    loss
      ↓
    Softmax
      ↓
    Dense 2
      ↓
    ReLU
      ↓
    Dense 1

Each layer receives a gradient from the layer after it.

It then calculates:
    1. How its inputs affected the loss
    2. How its weights affected the loss
    3. How its biases affected the loss

These are stored as:
    dinputs
    dweights
    dbiases
'''

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs # Save inputs because we will need them during backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        '''
        dvalues contains the gradient coming from the layer AFTER us.

        We need to calculate:
            dLoss/dWeights
            dLoss/dBiases
            dLoss/dInputs
        '''

        # -------------------------------------------------
        # Gradient with respect to weights
        # -------------------------------------------------
        '''
        In P11 we learned:derivative wrt weight = input

        But now we have many samples and many neurons.

        Matrix multiplication lets us calculate all of these
        gradients at once.
        '''

        self.dweights = np.dot(self.inputs.T, dvalues)

        # -------------------------------------------------
        # Gradient with respect to biases
        # -------------------------------------------------
        '''
        Bias is added directly to every neuron: output = ... + bias

        So: dOutput/dBias = 1

        We sum the gradients from every sample.
        '''

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


        # -------------------------------------------------
        # Gradient with respect to inputs
        # -------------------------------------------------
        '''
        In P11: derivative wrt input = weight

        Again, matrix multiplication performs this for
        every input and neuron at once.
        '''

        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs # Save inputs because we will need them for backward pass
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''
        ReLU derivative:
            input > 0  → derivative = 1
            input <= 0 → derivative = 0

        We start with the gradient coming from the next layer.
        '''

        self.dinputs = dvalues.copy()

        # Any neuron that had input <= 0 during forward pass gets gradient of 0.
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs # Save inputs because we will need them for backward pass

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        '''
        Softmax is more complicated than ReLU.

        Each Softmax output depends on EVERY input to Softmax.

        For example:
            probability 1 depends on:
                input 1
                input 2
                input 3

        Because all probabilities must add up to 1.

        Therefore we need something called a Jacobian matrix.
        '''

        # Empty array with same shape as incoming gradients
        self.dinputs = np.empty_like(dvalues)

        # Process each sample individually
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Convert row into column vector
            single_output = single_output.reshape(-1, 1)

            '''
            Softmax Jacobian:
                diag(output) - output * output.T
            '''

            jacobian_matrix = (np.diagflat(single_output) - np.dot(single_output, single_output.T))

            # Chain rule
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

    def backward(self, dvalues, y_true):
        '''
        Now we need: dLoss/dSoftmaxOutput

        For categorical cross entropy:
                 -y_true
            ----------------
                 y_pred

        If the correct class probability is low,
        the gradient will be larger.

        That means the network receives a stronger signal
        that the prediction was bad.
        '''

        samples = len(dvalues)

        labels = len(dvalues[0])

        # If labels are scalar:
        # [0, 1, 2]
        # Convert them to one-hot form.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate Gradient and normalize it (This keeps the gradient magnitude roughly independent of num samples)
        self.dinputs = (-y_true / dvalues) / samples



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





# =========================================================
# FORWARD PASS
# =========================================================
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y) 

print("Loss:", loss)
print("\nFirst 5 predictions:")
print(activation2.output[:5])




# =========================================================
# BACKWARD PASS
# =========================================================
'''
Now we begin at the end of the network and work backward.

Forward:
    dense1
      ↓
    ReLU
      ↓
    dense2
      ↓
    Softmax
      ↓
    Loss

Backward:
    Loss
      ↓
    Softmax
      ↓
    dense2
      ↓
    ReLU
      ↓
    dense1
'''

# Loss backward
loss_function.backward(activation2.output, y)

# Softmax backward
activation2.backward(loss_function.dinputs)

# Dense 2 backward
dense2.backward(activation2.dinputs)

# ReLU backward
activation1.backward(dense2.dinputs)

# Dense 1 backward
dense1.backward(activation1.dinputs)



# =========================================================
# LOOK AT THE GRADIENTS
# =========================================================
print("\n-------------------------")
print("Dense 2 gradients")
print("-------------------------")

print("\ndWeights:")
print(dense2.dweights)

print("\ndBiases:")
print(dense2.dbiases)
print("\n-------------------------")

print("Dense 1 gradients")
print("-------------------------")

print("\ndWeights:")
print(dense1.dweights)

print("\ndBiases:")
print(dense1.dbiases)

print("\nGradient shapes:")
print(
    "Dense1 weights:",
    dense1.weights.shape,
    "→ dweights:",
    dense1.dweights.shape
)

print(
    "Dense1 biases:",
    dense1.biases.shape,
    "→ dbiases:",
    dense1.dbiases.shape
)

print(
    "Dense2 weights:",
    dense2.weights.shape,
    "→ dweights:",
    dense2.dweights.shape
)

print(
    "Dense2 biases:",
    dense2.biases.shape,
    "→ dbiases:",
    dense2.dbiases.shape
)


'''
Summary
---------------------------------------
We now have a complete forward and backward pass.

FORWARD:
    X
    ↓
    Dense 1
    ↓
    ReLU
    ↓
    Dense 2
    ↓
    Softmax
    ↓
    Cross Entropy Loss

BACKWARD:
    Loss.backward()
    ↓
    Softmax.backward()
    ↓
    Dense2.backward()
    ↓
    ReLU.backward()
    ↓
    Dense1.backward()


After backpropagation, our Dense layers now contain:
    dense1.dweights
    dense1.dbiases

    dense2.dweights
    dense2.dbiases

These gradients tell us:
    "If I change this parameter,
     how does the loss change?"

For example:
    dense1.dweights[0][0]

represents approximately:
    dLoss
    --------
    dWeight[0][0]

We finally have the information that our random optimizer
in P9 was missing.

P9:
    Randomly change weight
    ↓
    Calculate loss
    ↓
    Did it help?
    ↓
    Maybe keep it

Now:
    Calculate gradient
    ↓
    Know which direction affects loss
    ↓
    Update weight intentionally

However, THIS FILE DOES NOT UPDATE THE WEIGHTS YET.

It only calculates the gradients.

Next:
    P14 - Stochastic Gradient Descent (SGD)

We will use:
    dweights
    dbiases

to actually update:
    weights
    biases
and train the network over many iterations.
'''