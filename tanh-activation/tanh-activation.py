import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    return np.tanh(np.array(x))
# Test 1
x1 = [0, 1, -1,31]
print("Test 1:", tanh(x1))

# Test 2
x2 = [[0,1],[-1,2]]
print("Test 2:", tanh(x2))

