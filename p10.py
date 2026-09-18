import numpy as np


'''
Overview
---------------------------------------
In P9, we randomly changed weights and biases and kept the changes
if they lowered the loss.

That worked, but it was extremely inefficient because we had no idea
which direction each parameter should move.

P10 introduces derivatives.

A derivative tells us how much a function's output changes when we make
a very small change to its input.

For neural networks, this will eventually let us answer:

    "If I slightly change this weight, what happens to the loss?"

If increasing a weight increases the loss, we probably want to move
the weight in the opposite direction.

If increasing a weight decreases the loss, that direction may help us.

Before applying this to a neural network, we will start with a very
simple function.
'''

# ---------------------------------------------------------
# Simple function
# ---------------------------------------------------------

def f(x):
    return 2 * x

x = 1.0
print("f(x):", f(x))

'''
Our function is:
    f(x) = 2x

So if:
    x = 1

then:
    f(1) = 2

    
If:
    x = 2
then:
    f(2) = 4

The output increases twice as fast as x.
That means the slope, or derivative, is 2.
'''

# ---------------------------------------------------------
# Numerical derivative
# ---------------------------------------------------------
x = 1.0

# A very small change in x
h = 0.001

# Calculate the output before changing x
y1 = f(x)

# Calculate the output after changing x slightly
y2 = f(x + h)

print("\ny1:", y1)
print("y2:", y2)

# Change in output divided by change in input
derivative = (y2 - y1) / h
print("Derivative:", derivative)

'''
Derivative formula:

          change in y
    -----------------------
          change in x

or:

    f(x + h) - f(x)
    -----------------
            h

where h is a very small number.

For:
    f(x) = 2x

we get approximately:
    derivative = 2

This means that for every change of 1 in x,
the function changes by about 2.
'''

# ---------------------------------------------------------
# A slightly more interesting function
# ---------------------------------------------------------
def f2(x):
    return x**2

x = 2.0
h = 0.001

y1 = f2(x)
y2 = f2(x + h)

derivative = (y2 - y1) / h

print("\n-------------------------")
print("Function: x^2")
print("-------------------------")

print("x:", x)
print("f(x):", y1)
print("f(x + h):", y2)
print("Derivative:", derivative)

'''
For:
    f(x) = x^2

the derivative changes depending on where we are.

At x = 2:
    derivative ≈ 4

At x = 3:
    derivative ≈ 6

At x = 4:
    derivative ≈ 8

So unlike f(x) = 2x, this function does not have the same slope
everywhere.

This matters for neural networks because changing a weight may have
a different effect depending on the current values of the network.
'''

# ---------------------------------------------------------
# Try several x values
# ---------------------------------------------------------
print("\n Different x values:")

for x in range(-5, 6):
    x = float(x)

    y1 = f2(x)
    y2 = f2(x + h)

    derivative = (y2 - y1) / h

    print("x:", x, "f(x):", y1, "Derivative:", derivative)
'''
Notice:

    x = -2 -> derivative ≈ -4
    x =  0 -> derivative ≈  0
    x =  2 -> derivative ≈  4

The sign of the derivative tells us the direction of the slope.

Positive derivative:
    Increasing x increases the output.

Negative derivative: Increasing x decreases the output.

Derivative near zero: The function is relatively flat at that point.

This idea becomes extremely important during neural network training.

Eventually "x" will be replaced by something like "weight"
and: f(x) will effectively represent: loss

We want to know: "How does the loss change when this weight changes?"

That value is the derivative of the loss with respect to the weight.
'''

# ---------------------------------------------------------
# Another example: x^3
# ---------------------------------------------------------
def f3(x):
    return x ** 3

print("\n-------------------------")
print("Function: x^3")
print("-------------------------")

for x in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    y1 = f3(x)
    y2 = f3(x + h)

    derivative = (y2 - y1) / h

    print(
        "x:",
        x,
        "f(x):",
        y1,
        "derivative:",
        derivative
    )

'''
Summary
---------------------------------------

A derivative tells us how sensitive an output is to a small change
in an input.

Numerically, we can estimate it with:
    derivative = (f(x + h) - f(x)) / h
where h is a very small number.

For neural networks, we eventually want things like:
    dLoss
    -----
    dWeight

which means:
    "How much does the loss change when this weight changes?"

We also want:
    dLoss
    -----
    dBias

Once we know these values, we no longer have to randomly guess
how to change the network.

Instead, we can use the slope to determine which direction should
reduce the loss.

Next: Partial derivatives and multiple inputs.

A neuron does not have just one input.

It has multiple inputs:
    x1 * w1
    x2 * w2
    x3 * w3
    ...
    + bias

So in P11, we need to figure out how changing ONE input or ONE weight
affects the output while everything else stays the same.
'''