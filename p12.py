import numpy as np

'''
Overview
---------------------------------------
In P11, we learned how to calculate partial derivatives for a neuron.

For example:
    output = x * w + b

We learned:
    dOutput/dWeight = x
    dOutput/dInput  = w
    dOutput/dBias   = 1


But this only tells us how the weight affects the NEURON OUTPUT.

During training, we actually care about:
    How does the weight affect the LOSS?


The network contains multiple operations:
    weight
      ↓
    neuron output
      ↓
    activation
      ↓
    loss


Each operation affects the next one.

The chain rule lets us combine these derivatives.

If:
    weight affects neuron output

and:
    neuron output affects loss

then:
    weight also affects loss


The derivatives are multiplied together.
'''

# ---------------------------------------------------------
# Simple example
# ---------------------------------------------------------

# Lets start with two simple functions
# z = 2x
# y = z^2
# Full path: x -> z=2x -> y=z^2

def function_z(x):
    return 2 * x


def function_y(z):
    return z ** 2

x = 3.0

z = function_z(x)
y = function_y(z)

print("x:", x)
print("z:", z)
print("y:", y)


'''
At x = 3:
    z = 2 * 3 = 6
Then:
    y = 6^2 = 36
'''

# ---------------------------------------------------------
# Derivative of z with respect to x
# ---------------------------------------------------------

'''
z = 2x

So:
    dz/dx = 2
'''

dz_dx = 2.0

print("\ndz/dx:", dz_dx)

# ---------------------------------------------------------
# Derivative of y with respect to z
# ---------------------------------------------------------

'''
y = z^2

Derivative:
    dy/dz = 2z

Since:
    z = 6

we get:

    dy/dz = 12
'''

dy_dz = 2 * z

print("dy/dz:", dy_dz)

# ---------------------------------------------------------
# Chain rule
# ---------------------------------------------------------

'''
Now we multiply the derivatives:
    dy/dx = dy/dz * dz/dx
'''

dy_dx = dy_dz * dz_dx

print("dy/dx:", dy_dx)

'''
We get:
    dy/dx = 12 * 2
          = 24


This means that at x = 3, changing x slightly changes y
at a rate of about 24.
'''

# ---------------------------------------------------------
# Verify numerically
# ---------------------------------------------------------

def full_function(x):
    z = 2 * x
    y = z ** 2
    return y

h = 0.001

y1 = full_function(x)
y2 = full_function(x + h)

numerical_derivative = (y2 - y1) / h

print("\nNumerical derivative:", numerical_derivative)
print("Chain rule derivative:", dy_dx)
'''
These should be very close.

The numerical derivative may look like:
    24.004...

while the analytical derivative is:
    24

The tiny difference comes from using h = 0.001.
'''


# ---------------------------------------------------------
# Bring this closer to a neuron
# ---------------------------------------------------------

'''
Now let's use something that looks more like a neural network.

We will have:
    input
      ↓
    weight
      ↓
    neuron output
      ↓
    loss

For simplicity:
    neuron_output = input * weight
and:
    loss = neuron_output^2

So:
    weight
      ↓
    neuron_output
      ↓
    loss


We want: dLoss/dWeight
'''

input_value = 3.0
weight = 0.5

def neuron(input_value, weight):
    return input_value * weight

def loss_function(neuron_output):
    return neuron_output ** 2

neuron_output = neuron(input_value, weight)
loss = loss_function(neuron_output)

print("\n-------------------------")
print("Neural network example")
print("-------------------------")

print("Input:", input_value)
print("Weight:", weight)
print("Neuron output:", neuron_output)
print("Loss:", loss)

# ---------------------------------------------------------
# Step 1:
# derivative of neuron output with respect to weight
# ---------------------------------------------------------
# Neuron: neuron_output = input * weight
# From P11: dNeuronOutput/dWeight = input

dneuron_dweight = input_value

print("\ndNeuron/dWeight:", dneuron_dweight)

# ---------------------------------------------------------
# Step 2:
# derivative of loss with respect to neuron output
# ---------------------------------------------------------

# Loss: loss = neuron_output^2
# Derivative: dLoss/dNeuronOutput = 2 * neuron_output

dloss_dneuron = 2 * neuron_output
print("dLoss/dNeuron:", dloss_dneuron)

# ---------------------------------------------------------
# Step 3:
# chain rule
# ---------------------------------------------------------

# To find: dLoss/dWeight
# we multiply: dLoss/dNeuron * dNeuron/dWeight

dloss_dweight = dloss_dneuron * dneuron_dweight

print("dLoss/dWeight:", dloss_dweight)

'''
Using our values:
    input = 3
    weight = 0.5

Neuron output:
    3 * 0.5 = 1.5

Loss:
    1.5^2 = 2.25


Derivative of loss with respect to neuron output:
    2 * 1.5 = 3

Derivative of neuron output with respect to weight:
    3

Therefore:
    dLoss/dWeight = 3 * 3
                  = 9
'''

# ---------------------------------------------------------
# Numerical check
# ---------------------------------------------------------

def full_neural_function(weight):
    neuron_output = input_value * weight
    loss = neuron_output ** 2
    return loss

original_loss = full_neural_function(weight)

new_weight = weight + h
new_loss = full_neural_function(new_weight)

numerical_dloss_dweight = (
    new_loss - original_loss
) / h


print("\nNumerical dLoss/dWeight:", numerical_dloss_dweight)
print("Chain rule dLoss/dWeight:", dloss_dweight)

# ---------------------------------------------------------
# Add an activation function
# ---------------------------------------------------------
'''
Now let's add another operation.
    weight
      ↓
    neuron output
      ↓
    ReLU
      ↓
    loss

This is much closer to a neural network.
'''

def relu(x):
    return max(0, x)

input_value = 2.0
weight = 1.5

neuron_output = input_value * weight
activated_output = relu(neuron_output)
loss = activated_output ** 2

print("\n-------------------------")
print("With ReLU")
print("-------------------------")

print("Neuron output:", neuron_output)
print("ReLU output:", activated_output)
print("Loss:", loss)

# ---------------------------------------------------------
# Derivative of ReLU
# ---------------------------------------------------------

'''
ReLU:
    ReLU(x) = x, if x > 0
    ReLU(x) = 0, if x <= 0

Derivative:
    1 if input > 0
    0 if input <= 0


Since neuron_output is positive here:
    dReLU/dNeuron = 1
'''

if neuron_output > 0:
    drelu_dneuron = 1.0
else:
    drelu_dneuron = 0.0

# ---------------------------------------------------------
# Derivative of loss
# ---------------------------------------------------------
dloss_drelu = 2 * activated_output


# ---------------------------------------------------------
# Derivative of neuron with respect to weight
# ---------------------------------------------------------
dneuron_dweight = input_value

# ---------------------------------------------------------
# Full chain rule
# ---------------------------------------------------------

'''
Now the path contains three derivative steps:
    weight
      ↓
    neuron
      ↓
    ReLU
      ↓
    loss


So:
    dLoss/dWeight =
        dLoss/dReLU
        *
        dReLU/dNeuron
        *
        dNeuron/dWeight
'''

dloss_dweight = (dloss_drelu * drelu_dneuron * dneuron_dweight)

print("\ndLoss/dReLU:", dloss_drelu)
print("dReLU/dNeuron:", drelu_dneuron)
print("dNeuron/dWeight:", dneuron_dweight)

print("Final dLoss/dWeight:", dloss_dweight)

# ---------------------------------------------------------
# ReLU blocking the gradient
# ---------------------------------------------------------

'''
Now let's see what happens if the neuron output is negative.
'''

input_value = 2.0
weight = -1.5


neuron_output = input_value * weight
activated_output = relu(neuron_output)
loss = activated_output ** 2

if neuron_output > 0:
    drelu_dneuron = 1.0
else:
    drelu_dneuron = 0.0

dloss_drelu = 2 * activated_output
dneuron_dweight = input_value

dloss_dweight = (dloss_drelu * drelu_dneuron * dneuron_dweight)

print("\n-------------------------")
print("Negative ReLU example")
print("-------------------------")

print("Neuron output:", neuron_output)
print("ReLU output:", activated_output)
print("Loss:", loss)

print("dReLU/dNeuron:", drelu_dneuron)
print("dLoss/dWeight:", dloss_dweight)

'''
Because the neuron output is negative:
    ReLU output = 0

and:
    dReLU/dNeuron = 0


That zero gets multiplied through the chain:
    derivative * 0 * derivative = 0


So the gradient going backward through this ReLU becomes zero.

This is why ReLU blocks gradients for negative inputs.
'''

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

'''
Summary
---------------------------------------
The chain rule lets us calculate how something earlier in the network
affects something later in the network.

If:
    x → z → y

then:
    dy/dx = dy/dz * dz/dx

For a neural network:

    weight
      ↓
    neuron
      ↓
    activation
      ↓
    loss

we can calculate:
    dLoss/dWeight

by multiplying the derivatives along the path.

Example:
    dLoss/dWeight = dLoss/dActivation * dActivation/dNeuron * dNeuron/dWeight

This is the basic idea behind backpropagation.

Backpropagation starts at the loss and works backward through the network,
using the chain rule at every step.

Next:
    P13 will start implementing actual backward methods for neural-network
    layers and activations.
'''