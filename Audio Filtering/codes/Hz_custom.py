import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

# Read .wav file 
input_signal, fs = sf.read('Input_audio.wav') 

# Order of the filter
order = 3

# Cutoff frequency 4kHz
cutoff_freq = 4000.0 

# Digital frequency
Wn = 2 * cutoff_freq / fs 

# Compute the filter coefficients
b, a = signal.butter(order, Wn, 'low') 

w, h = signal.freqz(b, a, worN=8000)

plt.plot(w, np.abs(h),color = 'red')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|H(e^{j\omega})| $')
plt.grid()
plt.savefig("Filter_Response.png")
plt.show()

