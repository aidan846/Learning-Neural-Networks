# Learning Neural Networks

This repo builds a neural network from scratch in Python and NumPy, one step at a time. Each `pN.py` file adds one idea to the one before it. It starts with a single neuron doing plain arithmetic. It builds up to a classifier that uses backpropagation, the Adam optimizer, L2 regularization and dropout, and that is checked against validation data it never trained on. The last files reuse the same pieces for **regression**: predicting a continuous number (a sine wave) instead of a class.

P1–P8 follow the *Neural Networks from Scratch* (`nnfs`) video series. The series stops after P9, so P9 onward continue the same path without it: optimization, calculus, backprop, optimizers, regularization, validation and regression.

---

## Table of Contents

- [Setup](#setup)
- [The Big Picture](#the-big-picture)
- **Part 1: The forward pass**
  - [P1: A single neuron](#p1--a-single-neuron)
  - [P2: A layer of 3 neurons](#p2--a-layer-of-3-neurons)
  - [P3: Using NumPy and the dot product](#p3--using-numpy-and-the-dot-product)
  - [P4: Batches, multiple layers and the `Layer_Dense` class](#p4--batches-multiple-layers-and-the-layer_dense-class)
  - [P5: Activation functions (ReLU) and spiral data](#p5--activation-functions-relu-and-spiral-data)
  - [P6: Softmax](#p6--softmax)
  - [P6-2: A full forward pass with Softmax](#p6-2--a-full-forward-pass-with-softmax)
- **Part 2: Measuring error**
  - [P7: One-hot encoding and categorical cross-entropy](#p7--one-hot-encoding-and-categorical-cross-entropy)
  - [P8: The `Loss` classes](#p8--the-loss-classes)
- **Part 3: Learning**
  - [P9: Optimization by random search](#p9--optimization-by-random-search)
  - [P10: Derivatives](#p10--derivatives)
  - [P11: Partial derivatives](#p11--partial-derivatives)
  - [P12: The chain rule](#p12--the-chain-rule)
  - [P13: Backpropagation](#p13--backpropagation)
  - [P14: Stochastic Gradient Descent (SGD)](#p14--stochastic-gradient-descent-sgd)
  - [P15: Learning rate decay and momentum](#p15--learning-rate-decay-and-momentum)
  - [P16: AdaGrad, RMSprop and Adam](#p16--adagrad-rmsprop-and-adam)
  - [P17: Combined Softmax + cross-entropy backward pass](#p17--combined-softmax--cross-entropy-backward-pass)
- **Part 4: Generalization**
  - [P18: L1 / L2 regularization and visualization](#p18--l1--l2-regularization-and-visualization)
  - [P19: Dropout](#p19--dropout)
  - [P20: Validation data](#p20--validation-data)
- **Part 5: Regression**
  - [P21: Regression: learning a sine wave](#p21--regression-learning-a-sine-wave)
  - [P22: Regression with validation data](#p22--regression-with-validation-data)
- [plot.py: `SpiralVisualizer`](#plotpy--spiralvisualizer)
- [Cheat Sheet](#cheat-sheet)

---

## Setup

```bash
pip install numpy nnfs matplotlib
python p20.py
```

| Package | Used for |
|---|---|
| `numpy` | All the math: arrays, dot products, broadcasting |
| `nnfs` | `spiral_data()` for a toy dataset, and `nnfs.init()` for reproducibility |
| `matplotlib` | Decision-boundary plots (P18–P20) through `plot.py`, and live line plots of the predicted curve (P21–P22) |

**What `nnfs.init()` does:** it seeds NumPy's random number generator and sets the default data type to `float32`. That makes every run produce the same "random" weights and data, so results can be compared between files and with the course.

---

## The Big Picture

The finished network (P20) looks like this:

```
FORWARD  (make a prediction)
    X (x, y coordinates)
    ↓
    Dense 1  (2 → 100)
    ↓
    ReLU
    ↓
    Dropout (10%)                    ← training only
    ↓
    Dense 2  (100 → 3)
    ↓
    Softmax + Cross-Entropy Loss     → loss, accuracy

BACKWARD  (work out who is to blame)
    Softmax+Loss.backward → Dense 2 → Dropout → ReLU → Dense 1

UPDATE  (fix it a little)
    Adam optimizer nudges every weight and bias against its gradient

REPEAT  10,000 times
```

Every file adds one piece of this pipeline.

---

# Part 1: The forward pass

## P1: A single neuron

**Added:** the most basic unit of a neural network.

```python
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
```

**How a neuron works:**
- Each **input** is a number coming in, either raw data or the output of an earlier neuron.
- Each input has its own **weight**, which sets how much that input matters. A large positive weight means "this input pushes my output up a lot". A negative weight means "this input pushes my output down".
- The **bias** is a constant added at the end. It shifts the output up or down no matter what the inputs are, the way the `b` in `y = mx + b` shifts a line off the origin.

A neuron is a **weighted sum plus a bias**. The weights and biases are the network's **parameters**, and "training" means finding good values for them.

Result: `1.2·3.1 + 5.1·2.1 + 2.1·8.7 + 3 = 35.7`

---

## P2: A layer of 3 neurons

**Added:** several neurons that all read the **same inputs**, each with its own weights and bias.

```
inputs  = [1, 2, 3, 2.5]           (4 inputs)
neuron 1: weights1, bias1  → 4.8
neuron 2: weights2, bias2  → 1.21
neuron 3: weights3, bias3  → 2.385
output  = [4.8, 1.21, 2.385]       (3 outputs, one per neuron)
```

**Why several neurons?** Each neuron learns to detect something different in the same input. A layer of N neurons turns an input vector into an output vector of length N.

**Rule to remember:** each neuron has **one weight per input** and **one bias**. With 4 inputs and 3 neurons that is 3×4 = 12 weights and 3 biases.

(These three outputs, `[4.8, 1.21, 2.385]`, show up again as the first row of the example in P6.)

---

## P3: Using NumPy and the dot product

**Added:** P2 rewritten in one line.

```python
output = np.dot(weights, inputs) + biases
```

**What a dot product is:** multiply two lists element by element, then add up the results:
`[a, b, c] · [x, y, z] = a·x + b·y + c·z`. That is exactly what one neuron computes.

**Why `np.dot(weights, inputs)` gives the whole layer:** `weights` is a matrix of shape `(3, 4)`, with one row per neuron. Dotting a matrix with a vector takes the dot product of **each row** with the vector, so all 3 neurons are computed at once and the result has shape `(3,)`. Adding `biases` then adds each neuron's bias to its own output.

**Why the argument order matters:** `np.dot(weights, inputs)` is shape `(3,4)·(4,) → (3,)`. The other order, `(4,)·(3,4)`, fails because the inner dimensions (4 and 3) don't match.

**Why use NumPy at all:** it runs in optimized C code instead of Python loops. The commented-out loop version at the bottom of `p3.py` gives the same result, but much more slowly.

---

## P4: Batches, multiple layers and the `Layer_Dense` class

**Added:**
1. **Batches**: `X` is now a list of 3 samples, shape `(3, 4)`.
2. **Two stacked layers**: the output of layer 1 becomes the input of layer 2.
3. **The `Layer_Dense` class**, which is used in every later file.

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```

### Why batches?
Running many samples at once is much faster, because it is one big matrix multiplication. It also gives a more stable picture of how the network is doing than a single sample would.

### Why transpose? (the manual version)
In the commented-out manual version, `inputs` is `(3, 4)` (3 samples × 4 features) and `weights` is `(3, 4)` (3 neurons × 4 weights). Matrix multiplication needs the **inner dimensions to match**: `(a, b) · (b, c) → (a, c)`. A `(3,4)·(3,4)` product fails because 4 ≠ 3. Transposing the weights to `(4, 3)` makes it `(3,4)·(4,3) → (3,3)`, which is 3 samples × 3 neuron outputs. Now each sample's 4 features line up with each neuron's 4 weights.

### Why `Layer_Dense` doesn't need a transpose
The class creates its weights as `(n_inputs, n_neurons)` from the start, which is already the "transposed" shape. So `np.dot(inputs, self.weights)` works directly and no `.T` is needed on each forward pass.

### Why these starting values?
- **`np.random.randn`**: random numbers from a normal distribution centered on 0. The weights **must** start different from each other. If every weight were the same, every neuron would compute the same thing and receive the same updates, and they would never become different. This is called "breaking symmetry".
- **`× 0.10`**: keeps the starting weights small. Large starting weights produce very large outputs that grow layer after layer and make training unstable.
- **Biases start at 0**: this is a common, safe default, since the random weights already break symmetry.

### Chaining layers
```python
layer1 = Layer_Dense(4, 5)   # 4 inputs → 5 neurons
layer2 = Layer_Dense(5, 2)   # MUST take 5 inputs, because layer1 outputs 5 values
```
Each layer's `n_inputs` must equal the previous layer's `n_neurons`.

---

## P5: Activation functions (ReLU) and spiral data

**Added:** `Activation_ReLU` and the `spiral_data` dataset.

```python
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
```

### What ReLU is
**Re**ctified **L**inear **U**nit: `ReLU(x) = max(0, x)`. Negative values become 0 and positive values pass through unchanged.

### Why activation functions are needed
Without them, stacking layers does nothing useful. A dense layer is a linear function, and **a linear function of a linear function is still linear**. So 100 dense layers without activations can only do what a single dense layer can do, which is draw straight lines. An activation function adds a **non-linearity**, and that is what lets the network learn curved and complex shapes, like a spiral.

### Why ReLU in particular
- It is very cheap to compute (one comparison).
- Its gradient is simple: 1 for positive inputs, 0 otherwise (see P12).
- It avoids the "vanishing gradient" problem of older activations like sigmoid, whose gradients get tiny for large inputs.
- Many ReLU neurons together, each "switching on" at a different point, can approximate almost any shape as a series of connected straight pieces.

### Spiral data
`spiral_data(samples=100, classes=3)` creates 300 points arranged in 3 interlocking spiral arms.
- `X` has shape `(300, 2)`: each point's (x, y) coordinates. That is why the first layer now takes **2 inputs**.
- `y` has shape `(300,)`: each point's class label, 0, 1 or 2.

Spirals are a good test because **no straight line can separate them**, so the network has to learn a non-linear boundary.

---

## P6: Softmax

**Added:** turning raw layer outputs into **probabilities**.

```python
exp_values  = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
```

### What Softmax does
It takes any list of numbers and turns it into a probability distribution: every value becomes positive and the values add up to 1.

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

### Why each step
1. **Exponentiate (`e^x`)**: makes every value positive, even negative ones, while **keeping the order**, so a bigger input still gives a bigger output. It also stretches the differences, so the largest value stands out more.
2. **Normalize (divide by the sum)**: scales the values so they add up to 1 and can be read as "how confident the network is in each class".

### Why not just use ReLU at the output?
ReLU outputs aren't probabilities: they don't add up to 1 and can all be 0. For classification we want the output to answer "how confident is the network in each class?".

### `axis=1, keepdims=True`
- `axis=1` sums **across each row**, so each sample is normalized on its own.
- `keepdims=True` keeps the result as a `(3, 1)` column instead of a flat `(3,)` array, so NumPy can **broadcast** the division: each row is divided by its own sum.

---

## P6-2: A full forward pass with Softmax

**Added:** `Activation_Softmax`, and the first complete network:
`X → Dense(2,3) → ReLU → Dense(3,3) → Softmax`

```python
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
```

### Why subtract the max? (overflow protection)
`e^x` grows extremely fast: `e^1000` is `inf` in floating point, which breaks everything downstream. Subtracting each row's largest value makes the largest input 0, so every exponent is ≤ 0 and every `e^x` is between 0 and 1. No overflow is possible.

**It doesn't change the answer.** Subtracting a constant `c` from every input just multiplies the top and bottom of the fraction by the same `e^(-c)`, which cancels out.

### Why the output layer has 3 neurons
There are 3 classes, so the network produces one probability per class. At this stage the weights are random, so every output is about `[0.33, 0.33, 0.33]`. The network is guessing.

---

# Part 2: Measuring error

## P7: One-hot encoding and categorical cross-entropy

**Added:** a way to score how **wrong** a prediction is.

### One-hot encoding
Class labels are turned into vectors with a single `1` at the correct position:
```
dog, lizard, cat
is dog   → [1, 0, 0]
is cat   → [0, 0, 1]
```
This matches the shape of the Softmax output, so the two can be compared position by position.

### Categorical cross-entropy
```
L = -Σ y_true_i · log(y_pred_i)
```
Every `y_true_i` is 0 except the correct class, so all the other terms vanish and this simplifies to:
```
L = -log(predicted probability of the correct class)
```
Example: `softmax = [0.7, 0.1, 0.2]`, target = class 0 → `L = -log(0.7) ≈ 0.357`.

### Why use `-log`?
| Confidence in the correct class | Loss |
|---|---|
| 1.0 (perfect) | 0 |
| 0.7 | 0.357 |
| 0.33 (random guess, 3 classes) | 1.0986 |
| 0.1 | 2.303 |
| → 0 | → ∞ |

- It is **0 when the network is perfectly right** and **grows without limit as it becomes confidently wrong**. Being confidently wrong is punished much harder than being unsure.
- It only depends on the probability of the correct class. Since Softmax outputs add up to 1, raising that probability automatically lowers the others.
- `log` here is the **natural log** (base *e*): it answers "*e* to what power gives *b*?". That makes it the natural partner of the `e^x` in Softmax, and their derivatives combine very neatly (see P17).

**Useful fact:** a network guessing randomly among 3 classes has a loss of about `-log(1/3) ≈ 1.0986`. If your starting loss is close to that, the setup is working as expected.

---

## P8: The `Loss` classes

**Added:** `Loss` (base class) and `Loss_CategoricalCrossentropy`, which compute the average loss over a whole batch.

```python
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
```

### Why a base class?
`calculate()` (averaging over samples) is the same for **every** loss function. Only `forward()` changes. Putting `calculate()` in a parent class means future loss types (for example regression losses in P21) inherit it for free. This is also where regularization loss is added in P18.

### Clipping: `np.clip(y_pred, 1e-7, 1 - 1e-7)`
If the network ever predicts exactly `0` for the correct class, `-log(0) = ∞` and the loss becomes `inf`, which then breaks the average. Clipping keeps every prediction strictly between 0 and 1. The upper clip keeps things symmetric, so a prediction of exactly 1 can't produce a loss of exactly 0 while a prediction of 0 is clipped.

### Two label formats
```python
if len(y_true.shape) == 1:   # labels as class indices:  [0, 1, 1]
    correct_confidences = y_pred_clipped[range(samples), y_true]
elif len(y_true.shape) == 2: # one-hot:  [[1,0,0],[0,1,0],[0,1,0]]
    correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
```
- **Class indices:** `y_pred[range(samples), y_true]` is NumPy "fancy indexing". It pairs row `i` with column `y_true[i]` and picks out each sample's correct-class probability in one step.
- **One-hot:** multiplying by the one-hot matrix zeros out every wrong class, and summing each row leaves only the correct probability.

Both give the same thing: a vector of "how confident was the network in the right answer" for each sample. Then `-log` and the mean are applied.

---

# Part 3: Learning

## P9: Optimization by random search

**Added:** the first attempt at **training**: change the parameters and keep the change if the loss went down.

```python
for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(*dense1.weights.shape)
    # ... same for every weight and bias ...
    # forward pass → loss
    if loss < lowest_loss:  keep the new parameters
    else:                   restore the best parameters
```

### How it works
1. Save the best parameters found so far.
2. Add a small random nudge (`0.05 × random noise`) to every weight and bias.
3. Run the network and compute the loss.
4. If the loss improved, keep the change. If not, go back to the saved parameters.

### Why `.copy()`?
NumPy arrays are **references**. `best = dense1.weights` would make `best` point to the *same* array, so changing `dense1.weights` would also change `best`, and restoring would do nothing. `.copy()` makes an independent snapshot.

### Why this is a dead end
The loss does go down, but **very slowly**. The nudges are blind: they don't know which direction helps. Even this tiny network has 21 parameters (6+3 in Dense 1, 9+3 in Dense 2). Real networks have millions. Guessing in millions of dimensions almost never finds a better direction.

**What's missing:** a way to know, for each parameter, *which direction lowers the loss and by how much*. That is what derivatives provide, and P10–P13 build up to them.

---

## P10: Derivatives

**Added:** measuring how sensitive an output is to its input.

```python
derivative = (f(x + h) - f(x)) / h     # h = 0.001
```

### What a derivative is
The **slope** of a function at a point: "if I increase `x` by a tiny amount, how much does `f(x)` change?"

| Function | Derivative | Meaning |
|---|---|---|
| `f(x) = 2x` | `2` everywhere | the output always changes twice as fast as x |
| `f(x) = x²` | `2x` | the slope depends on where you are: -4 at x=-2, 0 at x=0, 4 at x=2 |
| `f(x) = x³` | `3x²` | always ≥ 0, and flat at x=0 |

### What the sign tells you
- **Positive**: increasing x increases the output.
- **Negative**: increasing x decreases the output.
- **Near zero**: the function is flat at that point.

### Why this matters for neural networks
Replace `x` with **a weight** and `f(x)` with **the loss**. Then the derivative answers the question P9 couldn't: *"If I change this weight, does the loss go up or down, and by how much?"* To lower the loss, move the weight in the **opposite direction** of the derivative.

### Numerical vs. analytical
The `(f(x+h) - f(x)) / h` method is a **numerical approximation**. It gives `4.001` instead of exactly `4` for x² at x=2, because `h` isn't truly zero. It is useful for checking work, but in a real network it would need an extra forward pass **for every parameter**, which is far too slow. From P11 on, we use the exact **analytical** formulas instead.

> Tip: the "central difference" `(f(x+h) - f(x-h)) / (2h)` is noticeably more accurate. Frameworks use this kind of check, called "gradient checking", to test their backprop code.

---

## P11: Partial derivatives

**Added:** derivatives for functions with **several inputs**, like a real neuron.

```
output = x1·w1 + x2·w2 + x3·w3 + b
```

### What a partial derivative is
The derivative with respect to **one** variable while **holding everything else fixed**. Written `∂output/∂w1`.

### The key results (verified numerically in the file)
| Partial derivative | Value | Why |
|---|---|---|
| `∂output/∂w_i` | `x_i` (the input) | `w_i` is multiplied by `x_i`, so a change in `w_i` is scaled by `x_i` |
| `∂output/∂x_i` | `w_i` (the weight) | same reasoning, the other way around |
| `∂output/∂b` | `1` | the bias is added directly |

With `inputs = [1, 2, 3]` and `weights = [0.2, 0.8, -0.5]`:
- weight gradients = `[1, 2, 3]` = the inputs
- input gradients = `[0.2, 0.8, -0.5]` = the weights

**This symmetry is the core of backprop for dense layers.**
- **Weight** gradients use the **inputs**. That is why P13 saves `self.inputs` during the forward pass.
- **Input** gradients use the **weights**. That is how the gradient is passed back to the previous layer.

A **gradient** is just the list of all partial derivatives, one for every parameter.

### The remaining problem
This tells us how a weight affects the **neuron's output**, but what we care about is how it affects the **loss**, which is several steps later (activation → next layer → softmax → loss). To connect the two we need the chain rule.

---

## P12: The chain rule

**Added:** combining derivatives across a chain of operations.

```
If  x → z → y,  then  dy/dx = dy/dz · dz/dx
```

### Intuition
If `z` changes 2× as fast as `x`, and `y` changes 12× as fast as `z`, then `y` changes 2 × 12 = 24× as fast as `x`. The rates **multiply** along the path.

### Worked examples in the file
1. **Pure math:** `z = 2x`, `y = z²`, at x=3 → `dy/dx = (2z)·2 = 12·2 = 24`. Checked numerically (≈24.004).
2. **Neuron + loss:** `out = input·w`, `loss = out²` → `dLoss/dw = 2·out · input = 3·3 = 9`.
3. **Neuron + ReLU + loss:**
   ```
   dLoss/dW = dLoss/dReLU · dReLU/dNeuron · dNeuron/dW
   ```

### The ReLU derivative and "dead" gradients
```
dReLU/dx = 1  if x > 0
           0  if x ≤ 0
```
If a neuron's output was negative, ReLU turned it into 0, and its derivative is 0. Since everything in the chain is **multiplied**, that 0 wipes out the whole gradient: `anything × 0 = 0`. **ReLU blocks gradients for neurons that were "off"**. That is correct, because changing the weight slightly wouldn't change a 0 output. But if a neuron is off for *every* sample, it stops learning completely. This is known as a "dead ReLU".

### Why this is backpropagation
Backprop is the chain rule applied systematically: start at the loss and go **backward** through the network. Each layer takes the gradient from the layer after it, multiplies by its own local derivative, and passes the result back. No layer needs to know about the whole network, only its own derivative.

---

## P13: Backpropagation

**Added:** a `backward()` method on every class, and a full backward pass through the real network. **No parameter updates yet.** This file only computes the gradients.

Each layer's `backward(dvalues)` receives `dvalues`, which is **∂Loss/∂(this layer's output)** from the layer after it, and produces:
- `dinputs`: the gradient to pass to the previous layer
- `dweights`, `dbiases`: only for dense layers, since only they have parameters

### Dense layer backward
```python
self.dweights = np.dot(self.inputs.T, dvalues)          # (n_inputs, samples)·(samples, n_neurons)
self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)  # sum over samples
self.dinputs  = np.dot(dvalues, self.weights.T)         # (samples, n_neurons)·(n_neurons, n_inputs)
```
This is P11 in matrix form:
- **`dweights` uses the inputs.** `inputs.T · dvalues` computes `input × gradient` for every weight **and adds it up over all samples in the batch** in one step. The result has the same shape as `weights`, as a gradient must.
- **`dbiases` = sum of `dvalues`.** The bias's local derivative is 1, so its gradient is the incoming gradient added up over the batch.
- **`dinputs` uses the weights.** Each input fed into *every* neuron, so its gradient is the weighted sum of all those neurons' gradients.

**Shape-check trick:** if you forget which side gets the transpose, choose the arrangement that makes the shapes line up. `dweights` must match `weights`, and `dinputs` must match `inputs`.

**Why `forward()` now stores `self.inputs`:** because `dweights` needs them. This is why every class in P13 and after saves its inputs.

### ReLU backward
```python
self.dinputs = dvalues.copy()
self.dinputs[self.inputs <= 0] = 0
```
The gradient passes through unchanged wherever the input was positive, and is zeroed wherever it wasn't (P12). `.copy()` avoids changing the incoming array in place.

### Softmax backward: the Jacobian
Softmax is harder because **every output depends on every input**: raising one probability must lower the others so they still add up to 1. So the derivative isn't one number per input but a full matrix of "how output *i* changes when input *j* changes", called the **Jacobian**:

```
J = diag(s) - s · sᵀ         (s = this sample's softmax output as a column)
J_ij = s_i(δ_ij - s_j)       (δ_ij = 1 if i == j, else 0)
```
The file builds this matrix **once per sample** in a Python loop and multiplies it by that sample's incoming gradient. It is correct but slow. P17 removes it.

### Cross-entropy backward
```python
self.dinputs = (-y_true / dvalues) / samples
```
- The derivative of `-log(p)` is `-1/p`. Multiplying by one-hot `y_true` keeps it only for the correct class. A low-confidence correct class produces a **large** gradient, meaning a strong push to fix it.
- **Why divide by `samples`?** The loss is a *mean*, so each sample's share is `1/N`. This also keeps the gradient size the same regardless of batch size, so the same learning rate works whether you use 10 samples or 10,000.
- `np.eye(labels)[y_true]` turns class indices into one-hot rows: `np.eye(3)` is the identity matrix, and indexing it with `[0, 2, 1]` picks out those rows.

### Order of the backward pass
It is the forward pass **in reverse**, with each call receiving the previous call's `dinputs`:
```
loss.backward → softmax.backward → dense2.backward → relu.backward → dense1.backward
```

---

## P14: Stochastic Gradient Descent (SGD)

**Added:** `Optimizer_SGD` and the first real training loop. The network actually learns now.

```python
layer.weights -= learning_rate * layer.dweights
layer.biases  -= learning_rate * layer.dbiases
```

### How it works
The gradient points **uphill**, in the direction that *increases* the loss fastest. Taking a small step the **opposite** way goes downhill. Repeat this thousands of times and the loss drops.

Picture standing on a hilly landscape in fog: you can't see the valley, but you can feel which way the ground slopes under your feet, so you keep stepping downhill.

### The training loop
```
for each epoch:
    1. forward pass
    2. compute loss and accuracy
    3. backward pass (compute gradients)
    4. update parameters
```
An **epoch** is one full pass over the training data.

### Accuracy
```python
predictions = np.argmax(activation2.output, axis=1)   # index of the highest probability
accuracy = np.mean(predictions == y)                 # fraction that are correct
```
Loss and accuracy measure different things. Accuracy only asks "was the top guess right?". Loss also cares about *how confident* the network was. Two networks can have the same accuracy with very different losses.

### The learning rate
The step size. **Too small** and training crawls. **Too large** and it overshoots the minimum and bounces around or diverges. Here it is `1.0`, which is quite large but works for this small network with averaged gradients.

### Why "stochastic"?
Strictly, SGD updates using a random subset ("mini-batch") of the data each step, which makes each gradient a noisy estimate. We use the full dataset every step, which is technically "batch gradient descent", but the name SGD is used for the update rule either way.

---

## P15: Learning rate decay and momentum

**Added:** two improvements to SGD, and a bigger hidden layer (`Dense(2, 64)`).

### Learning rate decay
```python
current_lr = learning_rate / (1 + decay * iterations)
```
**Why:** early in training you are far from a good solution, so large steps help you cover ground. Later you are near the bottom of a valley, and large steps overshoot back and forth. Decay starts large and **gradually shrinks** the step size. With `decay=1e-3` the rate halves after 1,000 iterations and is at 1/11 after 10,000.

### Momentum
```python
update = momentum * previous_update - learning_rate * gradient
```
**How it works:** each update keeps part (here 90%) of the previous update and adds the current gradient on top. Think of a ball rolling downhill that builds up speed.

**Why:**
- **Speeds up** progress along directions where the gradient keeps pointing the same way.
- **Reduces zig-zagging** in narrow valleys, where the gradient flips back and forth: those flips cancel out in the running total.
- **Carries the optimizer through** flat areas and small bumps (local minima) where the current gradient alone is too small to make progress.

**Implementation detail:** `hasattr(layer, 'weight_momentums')` creates the momentum arrays the first time each layer is updated. They are stored **on the layer**, because each parameter needs its own history.

### The optimizer's three stages
```python
optimizer.pre_update_params()      # compute the decayed learning rate once per step
optimizer.update_params(dense1)    # update each layer
optimizer.update_params(dense2)
optimizer.post_update_params()     # iterations += 1
```
The learning rate is computed once per step, before any layer is updated, so every layer uses the same rate.

### Why 64 neurons?
A 3-neuron hidden layer doesn't have enough capacity to trace a 3-armed spiral. More neurons give the network more "bends" to work with (see ReLU in P5).

---

## P16: AdaGrad, RMSprop and Adam

**Added:** three **adaptive** optimizers, which give **each parameter its own effective learning rate**.

**The core idea:** divide each parameter's update by the (square root of the) size of its recent gradients.
- Parameters with large, frequent gradients get **smaller** steps, which calms them down.
- Parameters with small, rare gradients get **relatively larger** steps, which helps them learn.

`epsilon` (1e-7) is added to the denominator to avoid dividing by zero.

### AdaGrad (Adaptive Gradient)
```python
cache += gradient²
param -= lr * gradient / (sqrt(cache) + eps)
```
It keeps a **running sum** of squared gradients. Problem: the sum **only ever grows**, so the effective learning rate keeps shrinking and training can stall before it's finished.

### RMSprop (Root Mean Square Propagation)
```python
cache = rho * cache + (1 - rho) * gradient²
param -= lr * gradient / (sqrt(cache) + eps)
```
It fixes AdaGrad by using a **moving average** instead of a sum. With `rho = 0.9`, old gradients fade away, so the cache reflects *recent* behavior and the learning rate doesn't shrink to zero. It uses a much smaller base learning rate (0.001).

### Adam (Adaptive Moment Estimation)
It combines **momentum** (the direction) with **RMSprop** (the per-parameter step size):
```python
m = beta_1 * m + (1 - beta_1) * gradient        # momentum:  average gradient
v = beta_2 * v + (1 - beta_2) * gradient²       # cache:     average squared gradient
m_hat = m / (1 - beta_1 ** t)                   # bias correction
v_hat = v / (1 - beta_2 ** t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

**Why bias correction?** `m` and `v` start at zero, so for the first few steps they are pulled strongly toward zero. With `beta_1 = 0.9`, the very first `m` is only 10% of the real gradient. Dividing by `(1 - beta^t)` scales the early values back up to a fair estimate. As `t` grows, `beta^t → 0` and the correction fades away. (`t = iterations + 1` because `iterations` starts at 0.)

**Defaults:** `beta_1 = 0.9` (a short memory for direction), `beta_2 = 0.999` (a long memory for size).

### Summary
| Optimizer | Idea in one line |
|---|---|
| SGD | Follow the current gradient |
| SGD + Momentum | Follow the gradient, but keep your speed |
| AdaGrad | Slow down parameters that have seen a lot of gradient, ever |
| RMSprop | Slow down parameters that have seen a lot of gradient, recently |
| **Adam** | Momentum + RMSprop + bias correction |

Adam is the **most common default** in practice because it works well across many problems with little tuning. It is used for the rest of the project. (The file keeps AdaGrad and RMSprop commented out so you can swap them in and compare.)

---

## P17: Combined Softmax + cross-entropy backward pass

**Added:** `Activation_Softmax_Loss_CategoricalCrossentropy`, one object that does Softmax and the loss together and has a **much simpler backward pass**.

### The simplification
When Softmax is immediately followed by cross-entropy, multiplying their derivatives together cancels almost everything, and the result is:
```
∂Loss/∂z = ŷ - y
```
(the predicted probabilities minus the one-hot truth), divided by the number of samples.

**Why it cancels:** cross-entropy's derivative has a `1/ŷ` term, and the Softmax Jacobian has `ŷ` terms that cancel it out. Combined with the fact that the probabilities add up to 1, only `ŷ - y` is left.

### In code
```python
self.dinputs = dvalues.copy()                  # start with the predicted probabilities
self.dinputs[range(samples), y_true] -= 1      # subtract 1 at the correct class
self.dinputs = self.dinputs / samples
```
Example: prediction `[0.2, 0.7, 0.1]` with correct class 1 → gradient `[0.2, -0.3, 0.1]`.
- A wrong class gets a **positive** gradient, so its score is pushed down.
- The correct class gets a **negative** gradient, so its score is pushed up.
- A perfect prediction gives a gradient of **0**, so nothing changes.

### Why do it this way?
- **Faster:** no Jacobian matrix built in a per-sample Python loop.
- **Simpler:** one line of arithmetic.
- **More numerically stable:** no division by tiny probabilities.
- **Same result:** the forward math is unchanged. Only the gradient calculation is shortcut.

The learning rate is also raised to `0.05` for Adam here, which trains noticeably faster on this problem.

---

# Part 4: Generalization

## P18: L1 / L2 regularization and visualization

**Added:** weight penalties to fight **overfitting**, a larger hidden layer (`Dense(2, 100)`), and live decision-boundary plots through `plot.py`.

### The problem: overfitting
With enough neurons, a network can **memorize** the training points, drawing tight, jagged boundaries around each one, instead of learning the **general shape** of the spiral. It scores well on data it has seen and poorly on new data. Memorizing usually shows up as a few **very large weights**.

### The fix: penalize large parameters
```
total loss = data loss + regularization loss
```

**L1 regularization:** `λ · Σ|w|`
- Gradient: `λ · sign(w)`, a constant push toward zero no matter how big the weight is.
- Tends to push small weights to **exactly zero**, producing **sparse** networks (which also acts as a kind of feature selection).

**L2 regularization:** `λ · Σw²`
- Gradient: `2λw`, a push that is proportional to the weight's size.
- Large weights are punished hard, small ones barely at all. Weights rarely become exactly 0.
- Encourages spreading the work across **many small weights** instead of a few large ones. One weight of 10 costs 100, while ten weights of 1 cost only 10.
- **The more common choice.** Used here with `λ = 5e-4` on both layers.

### Why add it to the gradient too?
Adding the penalty to the printed loss only *reports* it. The optimizer only sees gradients, so the penalty's derivative must be added in `backward()`:
```python
self.dweights += 2 * self.weight_regularizer_l2 * self.weights
```
Now every update balances "fit the data" against "keep the weights small". (L2 used this way is often called **weight decay**.)

### Choosing λ
Too small and it has no effect. Too large and the network is so restricted it can't learn the pattern at all (**underfitting**). The goal isn't tiny weights, it's avoiding *needlessly extreme* ones.

### Visualization and a subtle bug fix
```python
if not epoch % 500:
    visualizer.update(epoch, model_predict)
    # re-run the forward pass on X ...
```
`model_predict()` sends **thousands of grid points** through the same layer objects, which **overwrites** their saved `self.inputs` and `self.output`. If we ran backward right after, the gradients would be calculated from the grid points instead of the training data, and the shapes wouldn't even match. So the training forward pass is **re-run** after every visualization to restore the correct state.

---

## P19: Dropout

**Added:** `Layer_Dropout`, a second, completely different regularization technique.

```
X → Dense(2,100) → ReLU → Dropout(0.1) → Dense(100,3) → Softmax+Loss
```

### What it does
On every training pass, it randomly **turns off** a fraction of neurons by setting their outputs to 0. A different random set is dropped every epoch.

### Why it works
Without dropout, the network can rely heavily on a few specific neurons, or on neurons that only work in particular combinations ("co-adaptation"). With dropout, **any neuron might disappear at any time**, so the network is forced to spread what it knows across many neurons. It is a bit like training many slightly different sub-networks and averaging them.

### How it works
```python
self.rate = 1 - rate                                             # store the KEEP rate (0.9)
self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
self.output = inputs * self.binary_mask
```
- `np.random.binomial(1, 0.9, shape)` draws a 1 with 90% probability and a 0 with 10% probability for each value, like a coin flip for every neuron.
- Multiplying by the mask zeroes out the dropped neurons.

### Inverted dropout: why divide by the keep rate?
If 10% of neurons are zeroed, the layer's total output is about 10% smaller during training than during prediction, when every neuron is active. The next layer would see differently sized inputs in the two cases. Dividing the mask by `0.9` scales the surviving neurons **up** by about 1.11×, so the **expected** output stays the same. Because this scaling happens during training, **prediction needs no adjustment at all**.

### Backward
```python
self.dinputs = dvalues * self.binary_mask
```
It uses the **same mask** as the forward pass: dropped neurons contributed nothing, so they get no gradient, and surviving neurons get their gradient scaled by the same 1/0.9.

### Training vs. prediction (`training=False`)
Dropout must be **off** during prediction, validation and visualization. Otherwise the model would give **different answers for the same input** each time it is called. `model_predict()` passes `training=False`. The visualizer fix from P18 also re-runs dropout, which creates a fresh training mask before backprop.

### Expect lower training accuracy
Dropout intentionally makes training **harder**, so training accuracy may be *lower* than in P18. That's fine: the goal is better performance on **unseen** data, which P20 finally measures. P19 uses a lighter L2 (`1e-4`) because dropout is now doing part of the regularizing.

---

## P20: Validation data

**Added:** a separate **validation set** the network never trains on, and a final evaluation that compares training and validation performance.

```python
X,     y     = spiral_data(samples=100, classes=3)   # training
X_val, y_val = spiral_data(samples=100, classes=3)   # validation: same spiral shape, new random points
```

### Why this matters
Training accuracy only shows how well the network fits data **it has already seen**. A network that memorized the training data could score 99% and still fail on new points. The real goal is **generalization**: learning the pattern, not the specific examples. The only honest way to measure that is with data the model has **never** been trained on.

### Rules for validation data
| | Training data | Validation data |
|---|---|---|
| Forward pass | ✅ | ✅ |
| Loss / accuracy | ✅ | ✅ |
| Backward pass | ✅ | ❌ |
| Parameter updates | ✅ | ❌ |
| Dropout | ✅ on | ❌ off |
| Regularization loss | ✅ included | ❌ excluded |

**Why validation loss excludes regularization:** regularization is a *training tool* for shaping the weights, not part of how good the predictions are. Validation should measure **prediction quality only**.

### Why evaluate the training data a second time?
The accuracy printed during training is measured **with dropout on**, so it describes a weakened version of the network. To compare training and validation fairly, the file runs the training data again with `training=False`, using the full network, just like the validation run.

### Reading the results
| Train | Validation | Diagnosis |
|---|---|---|
| 97% | 95% | ✅ Good generalization |
| 99% | 70% | ❌ **Overfitting**: memorized the training data |
| 65% | 64% | ❌ **Underfitting**: the model is too simple or undertrained |

The **generalization gap** (`train_acc - val_acc`) is printed at the end. A small gap is the sign of a healthy model.

### Visualization
Two `SpiralVisualizer`s show the **same learned decision boundary** over the training points and over the validation points. `plt.ioff()` followed by one `plt.show()` displays both final figures at the same time.

---

# Part 5: Regression

## P21: Regression: learning a sine wave

**Added:** `Activation_Linear` and `Loss_MeanSquaredError`. The network now predicts a **number** instead of a class.

```
X (one number, 0 to 1) → Dense(1,64) → ReLU → Dense(64,64) → ReLU → Dense(64,1) → Linear → MSE loss
```

```python
X = np.arange(0, 1, 0.005).reshape(-1, 1)   # 200 inputs: 0.000, 0.005, 0.010, ...
y = np.sin(2 * np.pi * X)                   # target: one full sine wave
```

### Classification vs. regression
| | Classification (P5–P20) | Regression (P21+) |
|---|---|---|
| Answer | Which class? (0, 1 or 2) | What number? (for example -0.71) |
| Output layer | One neuron per class | One neuron per value to predict |
| Output activation | Softmax (probabilities) | Linear (the raw number) |
| Loss | Categorical cross-entropy | Mean Squared Error |
| Progress measure | Accuracy | MSE / RMSE |

### Why `reshape(-1, 1)`?
`np.arange` gives a flat array of shape `(200,)`. Layers expect `(samples, features)`, so it is reshaped to `(200, 1)`: 200 samples with 1 feature each. `-1` tells NumPy to work out that dimension itself.

### Linear activation
```python
self.output = inputs            # forward: do nothing
self.dinputs = dvalues.copy()   # backward: the derivative of x is 1
```
Softmax squeezes outputs into probabilities between 0 and 1, and ReLU cuts off everything below 0, but a sine wave needs values from -1 to 1. The linear activation passes the last layer's raw output through unchanged. It exists mostly so every layer has the same `forward`/`backward` interface.

### Mean Squared Error (MSE)
```python
sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
```
- **Squaring** makes every error positive and punishes big misses much more than small ones: an error of 0.1 costs 0.01, an error of 1 costs 1.
- `axis=-1` averages over the outputs of each sample (there is only one here). `calculate()` then averages over all samples.

**Backward:** the derivative of `(y - ŷ)²` with respect to `ŷ` is `-2(y - ŷ)`:
```python
self.dinputs = -2 * (y_true - dvalues) / outputs   # divide by outputs: the forward pass averaged over them
self.dinputs = self.dinputs / samples              # divide by samples: same reason as P13/P17
```

### RMSE
`rmse = np.sqrt(loss)` undoes the squaring, so the error is back in the **same units as `y`**. An RMSE of `0.001` means predictions are off by roughly 0.001, which is easy to picture on a wave that goes from -1 to 1. An MSE like `1e-6` is much harder to read.

### Why a deeper network?
A ReLU network builds its output from straight line pieces joined at "kinks". Two hidden layers of 64 neurons give it many more pieces to bend into a smooth curve. Adam uses `learning_rate=0.005` and `decay=1e-3`: big steps early to find the shape, smaller steps later to fine-tune it.

### Visualization
Instead of `SpiralVisualizer`, a plain line plot shows the true sine wave and the network's current prediction. Every 100 epochs the prediction line is updated with `set_ydata()`, so you can watch it bend into shape. It starts flat and ends almost exactly on top of the sine wave. On a sample run, the final training RMSE was about **0.001**.

---

## P22: Regression with validation data

**Added:** the P20 idea applied to regression. P21 showed the network can fit the points it trained on. P22 asks: **can it predict the sine wave at points it never saw?**

```python
X     = np.arange(0, 1, 0.05).reshape(-1, 1)    # 20 training points (P21 had 200)
X_val = np.arange(0, 1, 0.001).reshape(-1, 1)   # 1,000 validation points
```

The network, loss and optimizer are the same as in P21.

### Why fewer training points?
With 200 closely spaced training points there is almost no gap to fill in, so a low training loss nearly guarantees a good curve. With only **20 points**, 0.05 apart, the network has to **interpolate**: guess what happens between the points. Checking it on a dense grid of 1,000 points shows whether it learned a smooth sine wave or just drew some line that happens to pass through the 20 dots.

### Evaluating after training
After the loop, the file runs two extra forward passes, like P20:
1. **Training data:** the final training MSE and RMSE.
2. **Validation data:** MSE and RMSE on the 1,000 points, with no backward pass and no updates.

```python
training_predictions = activation3.output.copy()
```
The validation pass reuses the same layer objects, so it **overwrites** `activation3.output`. `.copy()` saves the training predictions first. This is the same issue P18 ran into with the visualizer.

### Reading the results
On a sample run:

| | MSE | RMSE |
|---|---|---|
| Training (20 points) | `4.1e-10` | `0.00002` |
| Validation (1,000 points) | `3.4e-5` | `0.0058` |

The validation error is far larger than the training error, because the network hits its 20 training points almost exactly. In absolute terms, though, an RMSE of about 0.006 on a wave from -1 to 1 is still a close fit, so the network learned the shape of the curve and not only the points. Most of the validation error comes from the gaps **between** the training points and from the stretch past the last one (`X` stops at 0.95, `X_val` goes to 0.999), where the network has to **extrapolate**.

### Visualization
The live plot during training shows only the 20 training points. At the end, a new figure shows three things together:
- the **training points** (a scatter of what the network actually saw),
- the **actual function** over the dense validation grid,
- the **model's prediction** over that same grid.

Wherever the prediction line leaves the true curve, the network guessed wrong between the points it was given.

---

## plot.py: `SpiralVisualizer`

A helper used by P18–P20 to draw the network's **decision boundaries**, the regions of the plane it assigns to each class.

**How it works:**
1. Builds a fine **grid of (x, y) points** covering the data (`np.meshgrid`), then flattens it into an `(N, 2)` array (`np.c_[xx.ravel(), yy.ravel()]`), which is the same shape as `X`.
2. Sends the whole grid through the model with `predict_func` to get a class for every grid point.
3. Reshapes the predictions back into the grid and draws them as colored regions (`contourf`), with the real data points on top (`scatter`).

**Design choices:**
- `update()` uses a **coarse** grid (step 0.05) for fast live updates during training, and `show_final()` uses a **fine** grid (step 0.01) for a clean final image.
- `plt.ion()` (interactive mode) plus `plt.pause(0.001)` lets the plot refresh without stopping the training loop.
- `show_final()` returns the figure **without** calling `plt.show()`, so several final figures can be created and shown together (P20).
- It takes a **function**, not the model itself, so it works with any model that maps `(N, 2)` points to class labels.

---

## Cheat Sheet

| Concept | Forward | Backward (local derivative) |
|---|---|---|
| Dense | `X·W + b` | `dW = Xᵀ·dv`, `db = Σdv`, `dX = dv·Wᵀ` |
| ReLU | `max(0, x)` | `dv` where `x > 0`, else `0` |
| Softmax | `e^(x-max) / Σe^(x-max)` | Jacobian `diag(s) - s·sᵀ` |
| Cross-entropy | `-log(ŷ_correct)` | `-y / ŷ` / N |
| Softmax + CE | (both) | `(ŷ - y) / N` |
| Dropout | `x · mask / keep` | `dv · mask / keep` |
| L1 | `λΣ|w|` | `λ·sign(w)` |
| L2 | `λΣw²` | `2λw` |
| Linear | `x` | `dv` |
| MSE | `mean((y - ŷ)²)` | `-2(y - ŷ) / outputs / N` |

| Optimizer | Update |
|---|---|
| SGD | `w -= lr·g` |
| Momentum | `u = μu - lr·g;  w += u` |
| AdaGrad | `c += g²;  w -= lr·g/√c` |
| RMSprop | `c = ρc + (1-ρ)g²;  w -= lr·g/√c` |
| Adam | `m = β₁m + (1-β₁)g;  v = β₂v + (1-β₂)g²;  w -= lr·m̂/√v̂` |

**Shape rules:**
- `Layer_Dense(n_inputs, n_neurons)`: each layer's `n_inputs` must equal the previous layer's `n_neurons`.
- The first layer's `n_inputs` equals the number of features (2 for spiral data, 1 for the sine wave).
- The last layer's `n_neurons` equals the number of classes (3) for classification, or the number of values to predict (1) for regression.
- A gradient always has the **same shape** as the thing it is the gradient of.
