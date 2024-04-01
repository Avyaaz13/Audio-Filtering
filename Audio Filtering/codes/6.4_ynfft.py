import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft

# Read input audio file
input_signal, fs = sf.read('Input_audio.wav')

# Filter parameters
order = 3
cutoff_freq = 4000.0
Wn = 2 * cutoff_freq / fs
b, a = signal.butter(order, Wn, 'low')

# Length of impulse response
N = 278

# Generate impulse response
impulse = np.zeros(N)
impulse[0] = 1
h = signal.lfilter(b, a, impulse)

# Extract a portion of the input signal for processing
range_ = slice(1600, 1848)
xtemp = input_signal[range_]
x = np.pad(xtemp, (0, 30), 'constant', constant_values=(0))

# Compute FFT of x and h
X_fft = fft(x)
H_fft = fft(h)

# Multiply FFTs to perform convolution in frequency domain
Y_fft = H_fft * X_fft

# Compute the inverse FFT to obtain the output signal
y_ifft = ifft(Y_fft).real

# Calculate the output using scipy's lfilter
y_lfilt = signal.lfilter(b, a, x)[:N]

# Plot both output signals
plt.stem(range(0, N), y_lfilt, markerfmt='C1s',basefmt = 'k', linefmt='c-', label='sciPY')
plt.stem(range(0, N), y_ifft[:N], markerfmt='ro',basefmt = 'k', linefmt='g-', label='FFT')

plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.legend()
plt.grid()
plt.savefig('6.2_fft.png')
plt.show()

