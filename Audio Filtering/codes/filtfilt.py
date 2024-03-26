import soundfile as sf
from scipy import signal, fft
import numpy as np
from numpy.polynomial import Polynomial as P
from matplotlib import pyplot as plt

def custom_filtfilt(b, a, x):
    X = fft.fft(x)
    w = np.linspace(0, 1, len(X) + 1)
    W = np.exp(2j*np.pi*w[:-1])
    B = (np.absolute(np.polyval(b,W)))**2
    A = (np.absolute(np.polyval(a,W)))**2
    Y = B*(1/A)*X
    return fft.ifft(Y).real

x,fs = sf.read('Input_audio.wav') 

#order of the filter
order=3   

#cutoff frquency 4kHz
cutoff_freq=4000.0  

#digital frequency
Wn=2*cutoff_freq/fs 

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

#filter the input signal with butterworth filter
output_signal = signal.filtfilt(b, a, x)

y = custom_filtfilt(b, a, x)

x_plt = np.arange(len(x))

plt.stem(x_plt[1600:2000], output_signal[1600:2000],linefmt='g', markerfmt='go', basefmt='k', label='Scipy filtfilt')
plt.stem(x_plt[1600:2000], y[1600:2000],  linefmt='b-',markerfmt='ro', basefmt='k', label='Custom filtfilt')

plt.grid()
plt.legend()
plt.savefig("AudioFilter.png")

