import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft

input_signal, fs = sf.read('Input_audio.wav')

#order
order = 3

# Cutoff frequency 4kHz
cutoff_freq = 4000.0

# Digital frequency
Wn = 2 * cutoff_freq / fs
b, a = signal.butter(order, Wn, 'low')

N = 278
impulse = np.zeros(N)
impulse[0] = 1
h = signal.lfilter(b, a, impulse)

range_ = slice(1600,1848)

#xtemp=np.array([0.00634766, 0.00692749, 0.00686646, 0.00762939, 0.00671387, 0.00747681])
xtemp = input_signal[range_]
x=np.pad(xtemp, (0,30), 'constant', constant_values=(0))


# Perform signal convolution using np.convolve
y_convolve = np.convolve(x, h)

# Perform signal filtering using scipy.signal.lfilter
y_scipy = signal.lfilter(b , a , x)

plt.stem(range(0,N),y_scipy,markerfmt='go',basefmt = 'k',linefmt = 'ro-',label ='sciPY')
plt.stem(range(0, N), y_convolve[:N], markerfmt='ro',basefmt = 'k',linefmt = 'mo-', label='Convolution')


plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.legend()
plt.grid()
plt.savefig('6.2_yncon.png')
plt.show()

