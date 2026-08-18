import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    return 1 / (1 + np.exp(-np.array(x)))


# Test 1
x1 = [0, 2, -2]
print("Test 1:", sigmoid(x1))

# Test 2
x2 = 0
print("Test 2:", sigmoid(x2))

# Test 3
x3 = [[-1, 0], [1, 2]]
print("Test 3:", sigmoid(x3))