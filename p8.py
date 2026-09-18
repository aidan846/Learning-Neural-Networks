# Introducing Loss
# Calculate categorical cross-entropy loss to measure how wrong the network's predictions are

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



X, y = spiral_data(samples=100, classes=3) # Input is X,Y, so input is 3.

dense1 = Layer_Dense(2, 3) # 2 input features, 3 is output/# neurons
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3) # Prev output = 3, so this must be 3. And we have 3 classes so second num = 3
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y) 

print("Loss:", loss)




'''
import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1] 

# class_targets[0] checks softmax_outputs[0][0]
# class_targets[1] checks softmax_outputs[1][1]
# class_targets[1] checks softmax_outputs[2][1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]) # negative loss

average_loss = np.mean(neg_log)
print(average_loss)
'''