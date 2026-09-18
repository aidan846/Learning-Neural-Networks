import numpy as np

'''
Overview
---------------------------------------
In P10, we learned about derivatives using functions with one input.

For example: y = x^2

We estimated the derivative numerically using:
    derivative = (f(x + h) - f(x)) / h

This told us how much the output changed when x changed slightly.


Neural networks are more complicated because a neuron has multiple inputs
and multiple parameters.

For example: output = x1*w1 + x2*w2 + x3*w3 + bias

Now the output depends on:
    x1, x2, x3, ...
    w1, w2, w3, ...
    bias

So instead of asking: "How does the output change when x changes?"

we can ask: "How does the output change when w1 changes?"

while keeping everything else the same.

This is called a partial derivative.
'''


# ---------------------------------------------------------
# A single neuron
# ---------------------------------------------------------
inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 2.0

def neuron_output(inputs, weights, bias):
    output = (
        inputs[0] * weights[0] +
        inputs[1] * weights[1] +
        inputs[2] * weights[2] +
        bias
    )

    return output

output = neuron_output(inputs, weights, bias)

print("Original neuron output:", output)

'''
Our neuron is: output = x1*w1 + x2*w2 + x3*w3 + bias

Using our values: 
    output = 1.0*0.2  +  2.0*0.8  +  3.0*-0.5  +  2.0
    output = 0.2  +  1.6  +  -1.5  +  2.0
    output = 2.3
'''

# ---------------------------------------------------------
# Partial derivative with respect to weight 1
# ---------------------------------------------------------
h = 0.001

original_output = neuron_output(inputs, weights, bias)

# Copy the weights so we do not modify the originals
new_weights = weights.copy()

# Slightly increase weight 1
new_weights[0] += h

new_outputs = neuron_output(inputs, new_weights, bias)

derivative_w1 = (new_outputs - original_output) / h
print("\nDerivative with respect to w1:", derivative_w1)

'''
We only changed: w1

Everything else stayed the same. So this tells us:
    partial output
    --------------
    partial w1

Because: x1 = 1.0
The derivative is approximately 1.0

This makes sense because: x1 * w1
Is: 1.0 * w1

So if w1 increases by 1, the neuron output increases by 1.
'''

# ---------------------------------------------------------
# Partial derivative with respect to weight 2
# ---------------------------------------------------------
new_weights = weights.copy()

new_weights[1] += h

new_output = neuron_output(inputs, new_weights, bias)

derivative_w2 = (new_output - original_output) / h
print("Derivative with respect to w2:", derivative_w2)

'''
w2 is multiplied by: x2 = 2.0

So the derivative with respect to w2 is approximately: 2.0

If w2 changes slightly, its effect on the output is multiplied by x2.
'''

# ---------------------------------------------------------
# Partial derivative with respect to weight 3
# ---------------------------------------------------------
new_weights = weights.copy()

new_weights[2] += h

new_output = neuron_output(
    inputs,
    new_weights,
    bias
)


derivative_w3 = (
    new_output - original_output
) / h


print("Derivative with respect to w3:", derivative_w3)

'''
w3 is multiplied by: x3 = 3.0

So: dOutput/dw3 ≈ 3.0
'''

# ---------------------------------------------------------
# Partial derivative with respect to bias
# ---------------------------------------------------------
new_bias = bias + h


new_output = neuron_output(
    inputs,
    weights,
    new_bias
)


derivative_bias = (
    new_output - original_output
) / h

print("Derivative with respect to bias:", derivative_bias)
'''
The bias is simply added: output = ... + bias

So changing the bias by 1 changes the output by 1.

Therefore: dOutput/dbias = 1
'''

# ---------------------------------------------------------
# Compare all parameter derivatives
# ---------------------------------------------------------

print("\n-------------------------")
print("Parameter derivatives")
print("-------------------------")

print("dOutput/dw1:", derivative_w1)
print("dOutput/dw2:", derivative_w2)
print("dOutput/dw3:", derivative_w3)
print("dOutput/dbias:", derivative_bias)

'''
Notice something important:
    dOutput/dw1 = x1
    dOutput/dw2 = x2
    dOutput/dw3 = x3

For this neuron: inputs = [1.0, 2.0, 3.0]

so: weight derivatives = [1.0, 2.0, 3.0]

This is not a coincidence.

For: output = x*w + bias

the derivative with respect to the weight is:
    dOutput/dWeight = input

This will become very important when we implement backpropagation.
'''

# ---------------------------------------------------------
# Partial derivatives with respect to the inputs
# ---------------------------------------------------------
new_inputs = inputs.copy()
new_inputs[0] += h

new_output = neuron_output(
    new_inputs,
    weights,
    bias
)

derivative_x1 = (
    new_output - original_output
) / h


new_inputs = inputs.copy()
new_inputs[1] += h

new_output = neuron_output(
    new_inputs,
    weights,
    bias
)

derivative_x2 = (
    new_output - original_output
) / h


new_inputs = inputs.copy()
new_inputs[2] += h

new_output = neuron_output(
    new_inputs,
    weights,
    bias
)

derivative_x3 = (
    new_output - original_output
) / h

print("\n-------------------------")
print("Input derivatives")
print("-------------------------")

print("dOutput/dx1:", derivative_x1)
print("dOutput/dx2:", derivative_x2)
print("dOutput/dx3:", derivative_x3)

'''
Now notice:
    dOutput/dx1 = w1
    dOutput/dx2 = w2
    dOutput/dx3 = w3

For our neuron:
    weights = [0.2, 0.8, -0.5]

so the derivatives with respect to the inputs are approximately:
    [0.2, 0.8, -0.5]

Again, this makes sense.

For: x * w

the derivative with respect to x is: w

and the derivative with respect to w is: x
'''

# ---------------------------------------------------------
# Analytical version
# ---------------------------------------------------------

'''
So far, we estimated derivatives numerically by changing values by h.

But for this simple operation:
    output = x*w + bias

we already know the exact derivatives.

Weight derivatives:
'''

weight_derivatives = [
    inputs[0],
    inputs[1],
    inputs[2]
]

print("\nAnalytical weight derivatives:", weight_derivatives)


# Input derivatives:


input_derivatives = [
    weights[0],
    weights[1],
    weights[2]
]

print("Analytical input derivatives:", input_derivatives)

# Bias derivative:
bias_derivative = 1.0

print("Analytical bias derivative:", bias_derivative)

# ---------------------------------------------------------
# Vector form
# ---------------------------------------------------------

inputs_array = np.array(inputs)
weights_array = np.array(weights)


weight_derivatives_array = inputs_array
input_derivatives_array = weights_array


print("\nNumPy weight derivatives:", weight_derivatives_array)
print("NumPy input derivatives:", input_derivatives_array)


'''
Summary
---------------------------------------

For one neuron:
    output = x1*w1 + x2*w2 + x3*w3 + bias


The partial derivatives are:
    dOutput/dw1 = x1
    dOutput/dw2 = x2
    dOutput/dw3 = x3

So:
    dOutput/dWeights = inputs


And:
    dOutput/dx1 = w1
    dOutput/dx2 = w2
    dOutput/dx3 = w3

So:
    dOutput/dInputs = weights


The bias derivative is:
    dOutput/dbias = 1


This is useful, but we still have a problem.

We currently know how a weight affects the NEURON OUTPUT.

But our actual goal is:
    How does a weight affect the LOSS?


The path looks like:
    weight
      ↓
    neuron
      ↓
    activation
      ↓
    next layer
      ↓
    output
      ↓
    loss

To figure out how a change at the beginning affects something at the end,
we need to combine derivatives across multiple operations.

That is where the chain rule comes in.

P12: The chain rule and how gradients move backward through a network.
'''