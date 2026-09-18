import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from plot import SpiralVisualizer

nnfs.init()

'''
Overview
---------------------------------------
In P18, we introduced L1 and L2 regularization.

Regularization adds a penalty for large parameters and helps reduce
overfitting.

P19 introduces another regularization technique:

    Dropout

Dropout randomly disables some neurons during training.

For example, if we have:

    [Neuron 1]
    [Neuron 2]
    [Neuron 3]
    [Neuron 4]
    [Neuron 5]

Dropout might temporarily turn this into:

    [Neuron 1]  ON
    [Neuron 2]  OFF
    [Neuron 3]  ON
    [Neuron 4]  OFF
    [Neuron 5]  ON

On the next training pass, a different set may be disabled.

This prevents the network from depending too heavily on specific neurons.

IMPORTANT:

    Dropout is only used during training.

When testing, predicting, or visualizing the network, all neurons are used.
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

        # Store regularization strengths for this layer
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

        # L1 regularization gradients
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1

            self.dweights += self.weight_regularizer_l1 * dL1

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1

            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 regularization gradients
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

# ---------------------------------------------------------
# NEW: Dropout Layer
# ---------------------------------------------------------

class Layer_Dropout:
    def __init__(self, rate):
        '''
        rate is the fraction of neurons we want to drop
        
        Example: rate = 0.1 means we want to drop 10% of neurons
        '''

        self.rate = 1 - rate  # Invert rate for easier calculations

    def forward(self, inputs, training=True):
        self.inputs = inputs

        # -------------------------------------------------
        # NEW: Disable dropout when not training
        # -------------------------------------------------

        # During prediction/testing/visualization, dropout should NOT
        # randomly remove neurons.

        if not training:
            self.output = inputs.copy()
            return

        '''
        np.random.binomial() randomly generates either:

            0
            or
            1

        for every neuron output.

        If our keep rate is 0.9, each neuron has:

            90% chance of getting 1
            10% chance of getting 0

        Example mask:

            [1, 1, 0, 1, 0, 1]

        Multiplying the neuron outputs by this mask turns the
        zero positions off.
        '''

        self.binary_mask = np.random.binomial(
            1,
            self.rate,
            size=inputs.shape
        ) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # -------------------------------------------------
        # NEW: Backpropagate through dropout
        # -------------------------------------------------

        '''
        If a neuron was turned off during the forward pass, its mask was 0.

        Therefore its gradient should also be 0 during backpropagation.

        Neurons that survived dropout continue passing their gradients back.
        '''
        self.dinputs = dvalues * self.binary_mask


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

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums
            + (1 - self.beta_1) * layer.dweights
        )

        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums
            + (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = (
            layer.weight_momentums
            / (1 - self.beta_1 ** (self.iterations + 1))
        )

        bias_momentums_corrected = (
            layer.bias_momentums
            / (1 - self.beta_1 ** (self.iterations + 1))
        )

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache
            + (1 - self.beta_2) * layer.dweights ** 2
        )

        layer.bias_cache = (
            self.beta_2 * layer.bias_cache
            + (1 - self.beta_2) * layer.dbiases ** 2
        )

        weight_cache_corrected = (
            layer.weight_cache
            / (1 - self.beta_2 ** (self.iterations + 1))
        )

        bias_cache_corrected = (
            layer.bias_cache
            / (1 - self.beta_2 ** (self.iterations + 1))
        )

        # Final update weights and biases
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

X, y = spiral_data(samples=100, classes=3)


# ---------------------------------------------------------
# Network
# ---------------------------------------------------------

dense1 = Layer_Dense(
    2,
    100,
    weight_regularizer_l2=1e-4
)

activation1 = Activation_ReLU()


# ---------------------------------------------------------
# NEW: Dropout after the hidden activation
# ---------------------------------------------------------

'''
Dropout is normally placed AFTER an activation function.

Our network is now:
    X
    ↓
    Dense 1
    ↓
    ReLU
    ↓
    Dropout
    ↓
    Dense 2
    ↓
    Softmax + Loss

We're dropping 10% of the hidden activations.

That means on each training pass, roughly 90 of our 100 hidden neuron
outputs will remain active.
'''

dropout1 = Layer_Dropout(0.1)


dense2 = Layer_Dense(
    100,
    3,
    weight_regularizer_l2=1e-4
)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(
    learning_rate=0.05,
    decay=5e-7
)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

visualizer = SpiralVisualizer(X, y)


def model_predict(inputs):
    dense1.forward(inputs)
    activation1.forward(dense1.output)

    # NEW:
    # Dropout must be DISABLED during prediction.
    #
    # We want to use the full network when drawing the decision boundary,
    # not randomly remove neurons every time the visualizer asks for
    # predictions.
    dropout1.forward(activation1.output, training=False)

    dense2.forward(dropout1.output)
    loss_activation.activation.forward(dense2.output)

    return np.argmax(loss_activation.activation.output, axis=1)

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(10001):

    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    # -----------------------------------------------------
    # NEW: Dropout forward pass
    # -----------------------------------------------------

    '''
    Every epoch gets a newly-generated random dropout mask.

    This means the network is slightly different during every training pass.

    It cannot assume that any particular hidden neuron will always be
    available.
    '''

    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    # Data loss + regularization loss
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = (
        loss_activation.loss.regularization_loss(dense1)
        + loss_activation.loss.regularization_loss(dense2)
    )

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

            # Restore normal training forward-pass state after the visualizer.
            #
            # model_predict() calls our layers with visualization points,
            # replacing the saved inputs/output values used for backprop.
            # So we run the training data through the network again.
            dense1.forward(X)
            activation1.forward(dense1.output)

            # NEW:
            # We must also restore the training dropout state.
            # This creates a fresh training dropout mask before backprop.
            dropout1.forward(activation1.output)

            dense2.forward(dropout1.output)
            loss_activation.forward(dense2.output, y)

    # -----------------------------------------------------
    # Backward
    # -----------------------------------------------------

    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)

    # -----------------------------------------------------
    # NEW: Backpropagate through Dropout
    # -----------------------------------------------------

    '''
    Dense 2 gives us: dense2.dinputs

    These gradients first pass through dropout.

    Any neuron that was dropped during the forward pass also has its
    gradient dropped during the backward pass.
    '''

    dropout1.backward(dense2.dinputs)

    activation1.backward(dropout1.dinputs)
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

P18 introduced regularization by penalizing parameters:

    L1
    L2


P19 introduces:

    Dropout


DROPOUT
---------------------------------------

During training, Dropout randomly disables neuron outputs.

Our network:

    Dense 1
      ↓
    ReLU
      ↓
    Dropout
      ↓
    Dense 2


With:

    dropout rate = 0.1

we randomly disable approximately:

    10% of hidden neuron outputs

during each training pass.


WHY?
---------------------------------------

Without Dropout, the network might learn:

    "Neuron 42 is extremely useful.
     I will depend heavily on it."


But with Dropout:

    Epoch 1:
        neuron 42 exists

    Epoch 2:
        neuron 42 might disappear

    Epoch 3:
        neuron 42 exists again


So the network cannot depend completely on neuron 42.

It has to distribute what it learns across multiple neurons.


THE DROPOUT MASK
---------------------------------------

Suppose our activation output is:

    [2.0, 0.5, 1.2, 3.0]


A dropout mask might be:

    [1, 0, 1, 1]


Then:

    activation * mask

becomes:

    [2.0, 0.0, 1.2, 3.0]


The second neuron has temporarily disappeared.


INVERTED DROPOUT
---------------------------------------

Our actual mask does:

    mask / keep_rate


Why?

Suppose:

    dropout rate = 0.1
    keep rate = 0.9


If we simply removed 10% of neurons, the average output of the layer
would become smaller during training.

Instead, surviving neurons are slightly increased:

    1 / 0.9 ≈ 1.111


This keeps the expected output approximately the same during training
and prediction.

Because we do this DURING training, we do not need to scale anything
later during prediction.

This technique is called:

    inverted dropout


TRAINING VS PREDICTION
---------------------------------------

During training:

    dropout1.forward(output, training=True)

Dropout is active.


During prediction:

    dropout1.forward(output, training=False)

Dropout is disabled.


This is extremely important.

If Dropout stayed enabled during prediction, the model could give
different answers every time we called it.


BACKPROPAGATION
---------------------------------------

Forward:

    activation
        ↓
    dropout mask
        ↓
    next layer


Backward:

    next layer gradient
        ↓
    same dropout mask
        ↓
    previous activation


If a neuron was disabled during forward propagation:

    mask = 0

then its gradient during backpropagation is also:

    gradient = 0


REGULARIZATION + DROPOUT
---------------------------------------

Our current network uses both:

    L2 regularization
    +
    Dropout


These attack overfitting differently.

L2:

    discourages large weights


Dropout:

    discourages dependence on specific neurons


IMPORTANT
---------------------------------------

Do not necessarily expect training accuracy to improve.

Dropout intentionally makes training harder.

You may actually see LOWER training accuracy than P18.

What matters is whether the model performs better on data it has
never trained on.


That leads to the next important problem:

Right now we are only measuring:

    training accuracy


A model getting 97% on the same data it trained on does NOT prove
that it actually learned a general solution.


Next:
    P20 - Validation / testing with unseen data

We will generate a second spiral dataset that the network never trains on.

Then we can compare:

    training accuracy
        vs
    validation accuracy

and finally see whether regularization and dropout are actually helping.
'''