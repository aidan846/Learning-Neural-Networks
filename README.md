"# Learning-Neural-Networks" 

In P1, I learned how to add inputs, weights, and a bias into a singular output - 1 neuron.

In P2, I learned how to make 3 neurons, each with different weights and bias. And using the same inputs, they get combined into one output.

In P3, I learned how to simplify P2 using numpy, and using the dot() function to get the dot product of weights and input.

In P4, I learned how do 2 layers. And also why you have to transpose the weights -> this is because inputs is shape(3, 4), and weights is shape(3, 4) which means 3 neurons (rows) and 4 weights (columns), transpose changes weights (3,4) to (4,3). This is becasue if you have 3 neurons, each one has 4 weights. So the first array in weights (len 4) cannot be multiplied directly against the inputs because matrix multiplication requires the inner dimensions to match. By transposing weights into (4, 3), the 4 weights finally align with the 4 input features, allowing NumPy to successfully run the dot product and output a (3, 3) matrix for the layer."

In P5, I learned about activation functions, specifically Rectified Linear.