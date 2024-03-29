import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

# Read .wav file
input_signal, fs = sf.read('Input_audio.wav')

# Order of the filter
order = 3

# Cutoff frequency 4kHz
cutoff_freq = 4000.0

# Digital frequency
Wn = 2 * cutoff_freq / fs
b, a = signal.butter(order, Wn, 'low')

# Compute impulse response
impulse_length = 40  
impulse = np.zeros(impulse_length)
impulse[0] = 1
impulse_response = signal.lfilter(b, a, impulse)

plt.figure()
plt.stem(np.arange(len(impulse_response)), impulse_response, linefmt='b', markerfmt='ro', basefmt='k')
plt.xlabel('n')
plt.ylabel('h(n)')
plt.grid(True)
plt.savefig('h(n)_custom.png')

