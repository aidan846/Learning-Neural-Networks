'''
This example uses One-hot encoding

It convert categorical text data into a numerical binary format that algorithms can process. 
1 acts as a "yes" and 0 acts as "no". 

So, if there is the option for a dog, lizard, or cat

is_dog = 1
is_lizard = 0
is_cat = 0

So the target is = [1, 0, 0]

Next topic is Categorical Cross-Entropy
L_i = -log(y_i, k)  - L_i = loss, y_i is predicted values at an index(i), and k is the target/correct index

So for the previous example:
k = 0 because target[0] = 1. 

'''

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] + # you dont need target_output[0], because it = 1, 1 * any number = that number
		math.log(softmax_output[1]) * target_output[1] + # this ends up being 0 due to target_output[1] = 0
		math.log(softmax_output[2]) * target_output[2]) # this ends up being 0 due to target_output[2] = 0

print(loss)

loss = -math.log(softmax_output[0])
print(loss)



'''
# How logs work.
# solving for x
# e ** x = b

import numpy as np


b = 5.2

print(np.log(b))
'''