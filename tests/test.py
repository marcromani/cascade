import numpy as np

x = 2
y = -3
z = 5

f = (x + y) * z * x
print(f)

gradient = np.array([2 * x * z + y * z, x * z, x * x + x * y])
print(gradient)

sx = 1.5
sy = 2.5
sz = 1.2

xy = 0.5
xz = 1.8
yz = -1

matrix = np.array([[sx * sx, xy, xz], [xy, sy * sy, yz], [xz, yz, sz * sz]])
print(matrix)

print(gradient.reshape(1, 3) @ matrix @ gradient.reshape(3, 1))

A = np.array([[1, 2, 3], [0.1, 0.2, 0.3]])
B = np.array([[1, -1], [0, 1], [1, 0.5]])

print(A @ B)
