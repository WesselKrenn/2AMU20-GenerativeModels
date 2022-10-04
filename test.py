import numpy as np
import itertools
a = np.array([1,2,3]).T
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
states = np.array([state for state in itertools.product([0, 1], repeat=16)])
print(states)
print(states.shape)