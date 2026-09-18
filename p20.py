import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from plot import SpiralVisualizer
import matplotlib.pyplot as plt

nnfs.init()


'''
Overview
---------------------------------------
In P19, we introduced Dropout to help reduce overfitting.

But so far, we have only measured performance on the SAME data
that the network trained on.

That can be misleading.

A network might get:

    Training accuracy: 98%

but then get:

    Validation accuracy: 70%

on data it has never seen before.

That would mean the network learned the training samples very well,
but did not learn a solution that generalizes well.

P20 introduces:
    Training data
        and
    Validation data


Training data:

    Used for forward passes
    Used for backpropagation
    Used to update weights


Validation data:

    NEVER used for backpropagation
    NEVER used to update weights

It is only used to test the trained model.


This lets us compare:

    Training accuracy
        vs
    Validation accuracy
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

class Layer_Dropout:
    def __init__(self, rate):
        # rate is the fraction of neurons we want to drop.
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs

        # Dropout is disabled during testing/prediction.
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(
            1,
            self.rate,
            size=inputs.shape
        ) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
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
# NEW: Training and Validation Data
# ---------------------------------------------------------

'''
We now generate TWO datasets.

Training data:

    X, y

Validation data:

    X_val, y_val


The validation set comes from the same type of spiral_data() function,
but contains different randomly-generated points.

The model NEVER trains on X_val or y_val.
'''

X, y = spiral_data(samples=100, classes=3)

X_val, y_val = spiral_data(samples=100, classes=3)

# ---------------------------------------------------------
# Network
# ---------------------------------------------------------

dense1 = Layer_Dense(
    2,
    100,
    weight_regularizer_l2=1e-4
)

activation1 = Activation_ReLU()

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

# This visualizer continues showing the TRAINING dataset.
training_visualizer = SpiralVisualizer(X, y, title="Training Data")
validation_visualizer = SpiralVisualizer(X_val, y_val, title="Validation Data")


def model_predict(inputs):
    dense1.forward(inputs)
    activation1.forward(dense1.output)

    # Dropout is OFF for prediction/visualization.
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

    # Dropout is ON during training.
    dropout1.forward(activation1.output)

    dense2.forward(dropout1.output)

    # Data loss + regularization loss
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = (
        loss_activation.loss.regularization_loss(dense1)
        + loss_activation.loss.regularization_loss(dense2)
    )

    loss = data_loss + regularization_loss

    # Training accuracy
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
            training_visualizer.update(epoch, model_predict)

            # model_predict() overwrites the saved forward-pass state,
            # so restore the training forward pass before backpropagation.
            dense1.forward(X)
            activation1.forward(dense1.output)

            dropout1.forward(activation1.output)

            dense2.forward(dropout1.output)
            loss_activation.forward(dense2.output, y)

    # Backward
    loss_activation.backward(loss_activation.output, y)

    dense2.backward(loss_activation.dinputs)

    dropout1.backward(dense2.dinputs)

    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()


# ---------------------------------------------------------
# NEW: Final Training Evaluation
# ---------------------------------------------------------

'''
Our accuracy printed DURING training uses Dropout.

That is useful for seeing what the training process is doing, but it
doesn't represent how the finished model is actually used.

So now we evaluate the training data again with:

    training=False

This gives us the final training performance with the full network.
'''
dense1.forward(X)
activation1.forward(dense1.output)

dropout1.forward(activation1.output, training=False)

dense2.forward(dropout1.output)

training_loss = loss_activation.forward(dense2.output, y)

training_predictions = np.argmax(loss_activation.output, axis=1)
training_accuracy = np.mean(training_predictions == y)

# ---------------------------------------------------------
# NEW: Validation
# ---------------------------------------------------------

'''
Now we run the completely unseen validation data through the model.

IMPORTANT:

    No backward()
    No optimizer
    No parameter updates
    No Dropout


We are ONLY measuring how well the trained network performs.
'''

dense1.forward(X_val)
activation1.forward(dense1.output)

# Dropout is disabled during validation
dropout1.forward(activation1.output, training=False)

dense2.forward(dropout1.output)

validation_loss = loss_activation.forward(dense2.output, y_val)
validation_predictions = np.argmax(loss_activation.output, axis=1)
validation_accuracy = np.mean(validation_predictions == y_val)

generalization_gap = training_accuracy - validation_accuracy

# ---------------------------------------------------------
# NEW: Final Results
# ---------------------------------------------------------

print("\n-------------------------")
print("Final Results")
print("-------------------------")

print(
    "Training   -",
    "acc:", training_accuracy,
    "loss:", training_loss
)

print(
    "Validation -",
    "acc:", validation_accuracy,
    "loss:", validation_loss,
    "gap:", generalization_gap
)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

# Turn interactive mode off now that training is complete.
plt.ioff()

# Create both final figures without blocking.
training_visualizer.show_final(model_predict)
validation_visualizer.show_final(model_predict)

# Show both at the same time.
plt.show()





'''
Summary: Training vs. Validation
---------------------------------
- Training Data: Used to calculate loss, gradients, backpropagate, and update weights.
- Validation Data: Used ONLY to evaluate performance (forward pass, loss, accuracy). No backpropagation or weight updates.

Dropout During Validation
---------------------------------
- Must be disabled (training=False) so the entire network is used.

Interpreting Accuracy
---------------------------------
- 97% Train / 95% Val: Good generalization.
- 99% Train / 70% Val: Overfitting (memorized training data, poor on new data).
- 65% Train / 64% Val: Underfitting (failed to learn well).

Generalization
---------------------------------
- The ultimate goal is to learn patterns that work on new data, not memorize training data.

Why Validation Loss Excludes Regularization
---------------------------------
- Validation measures true prediction performance on unseen data.

Next: P21 - Regression
---------------------------------
- Moving from classification (x -> class) to predicting continuous values (x -> y).
'''