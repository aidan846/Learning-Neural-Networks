import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()


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

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples



class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = self.current_learning_rate = learning_rate
        self.decay, self.iterations = decay, 0
        self.epsilon, self.beta_1, self.beta_2 = epsilon, beta_1, beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            for p, name in ((layer.weights, 'weight'), (layer.biases, 'bias')):
                setattr(layer, f'{name}_momentums', np.zeros_like(p))
                setattr(layer, f'{name}_cache', np.zeros_like(p))

        t = self.iterations + 1
        for p_attr, d_attr, m_attr, c_attr in [
            ('weights', 'dweights', 'weight_momentums', 'weight_cache'),
            ('biases', 'dbiases', 'bias_momentums', 'bias_cache')
        ]:
            grad = getattr(layer, d_attr)
            momentums = self.beta_1 * getattr(layer, m_attr) + (1 - self.beta_1) * grad
            cache = self.beta_2 * getattr(layer, c_attr) + (1 - self.beta_2) * grad ** 2

            setattr(layer, m_attr, momentums)
            setattr(layer, c_attr, cache)

            m_corr = momentums / (1 - self.beta_1 ** t)
            c_corr = cache / (1 - self.beta_2 ** t)

            setattr(layer, p_attr, getattr(layer, p_attr) - self.current_learning_rate * m_corr / (np.sqrt(c_corr) + self.epsilon))

    def post_update_params(self):
        self.iterations += 1




X = np.arange(0, 1, 0.005).reshape(-1, 1)
y = np.sin(2 * np.pi * X)

# Network gets
# x = 0.00
# x = 0.005
# x = 0.010
# ...

# And is trying to learn: y = sin(2πx)

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()

optimizer = Optimizer_Adam(
    learning_rate=0.005,
    decay=1e-3
)


# Visualize
plt.ion()

fig, ax = plt.subplots()

actual_line, = ax.plot(X, y, label="Actual")
prediction_line, = ax.plot(X, np.zeros_like(y), label="Prediction")

ax.set_ylim(-1.5, 1.5)
ax.legend()

plt.show()


# Loop
for epoch in range(10001):

    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Loss
    loss = loss_function.calculate(activation3.output, y)

    if not epoch % 100:
        rmse = np.sqrt(loss)
        print(
            "epoch:", epoch,
            "mse:", loss,
            "rmse:", rmse,
            "lr:", optimizer.current_learning_rate
        )

        prediction_line.set_ydata(activation3.output)

        ax.set_title(
            f"Epoch: {epoch} | MSE: {loss:.8f} | RMSE: {rmse:.6f}"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(0.001)

    # Backward
    loss_function.backward(activation3.output, y)

    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)

    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    optimizer.post_update_params()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)
activation3.forward(dense3.output)

loss = loss_function.calculate(activation3.output, y)
rmse = np.sqrt(loss)

plt.ioff()

prediction_line.set_ydata(activation3.output)

ax.set_title(
    f"Final | MSE: {loss:.8f} | RMSE: {rmse:.6f}"
)

plt.show()