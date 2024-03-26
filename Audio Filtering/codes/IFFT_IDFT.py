import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

N = 14
n = np.arange(N)
fn=(-1/2)**n
hn1=np.pad(fn, (0,2), 'constant', constant_values=(0))
hn2=np.pad(fn, (2,0), 'constant', constant_values=(0))
h = hn1+hn2

x_temp=np.array([1.0,2.0,3.0,4.0,2.0,1.0])
x=np.pad(x_temp, (0,10), 'constant', constant_values=(0))

X_fft = fft(x)
H_fft = fft(h)
Y_fft = H_fft*X_fft
y_ifft = ifft(Y_fft).real

plt.stem(range(0,14),y_ifft[:14],markerfmt='ro',basefmt = 'k',linefmt = 'bo-')
plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.grid()
plt.legend(['IFFT'])
plt.savefig('yn_ver.png')
