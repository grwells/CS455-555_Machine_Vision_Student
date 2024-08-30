import numpy as np
import json

m = np.array([[1, 5, 5], [2, 6, 3]])
id = np.identity(3, dtype=np.float16)


print(id, m)

print(np.ones((3, 2)))
print(np.zeros((2,2)))

print(np.arange(0, 100).reshape(2,50))

b = m.tolist()

output_object = {
    'matrix': b
}

print(type(m.tolist()))

with open('output.json', 'w') as fp:
    json.dump(output_object, fp)
