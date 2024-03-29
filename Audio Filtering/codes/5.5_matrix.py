import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

N = 14
n = np.arange(N)
fn = (-1/2) ** n
hn1 = np.pad(fn, (0, 2), 'constant', constant_values=(0, 0))
hn2 = np.pad(fn, (2, 0), 'constant', constant_values=(0, 0))
h = hn1 + hn2

x_temp = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 1.0])
x = np.pad(x_temp, (0, 10), 'constant', constant_values=(0))
dftmatrix = fft(np.eye(len(x)))

X = x @ dftmatrix
H = h @ dftmatrix

# Convolution in the frequency domain
Y = H * X

# Inverse DFT matrix
invmatrix = np.linalg.inv(dftmatrix)
y = (Y @ invmatrix).real

plt.stem(range(0, 14), y[:14],markerfmt = 'ro',linefmt='b-',  basefmt='k')
plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.grid()
plt.savefig('matrix.png')

