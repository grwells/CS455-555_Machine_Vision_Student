import numpy as np
import json

# use numpy to create matrices
m = np.array([[1, 13, 5], [2, 6, 0.5]])
id = np.identity(3, dtype=np.float16)
ones = np.ones((3,3))

# reshaping from 2x3 -> 3x2
m = m.reshape((3,2))
print('m\n\t',m)

# simple math
# matrix product
print('matrix product\n\t', m.reshape((2,3))@id)
# matrix product
print('matrix product\n\t', m.reshape((2,3)).dot(id))
# element-wise
print('element/pairwise\n\t',id*ones)

# unary operations
print('m min val =', m.min())
print('m max val =', m.max())

# slicing
print(m[1, 1])

# create python dict
cam_matrix = {
    'random': np.array([1, 0.5, 0, 0, 1, 0]).reshape((2,3)).tolist(),
    'm': m.tolist()
}

# dict to string
print(json.dumps(cam_matrix))

# write to file
with open('filename_here.json', 'w') as fp:
	json.dump(cam_matrix, fp)
