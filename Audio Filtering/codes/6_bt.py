import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

#sampling frequency from the audio file
input_signal, fs = sf.read('Input_audio.wav')

T = 1.0/fs
#order of the filter
order=3

#cutoff frquency 
cutoff_freq=4000.0 

#digital frequency
Wn=2*cutoff_freq/fs


# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

#Bilinear Transform
def H(s):
	num = np.polyval(b,((1+s*(T/2))/(1-s*(T/2)))**(-1))
	den = np.polyval(a,((1+s*(T/2))/(1-s*(T/2)))**(-1))
	H = num/den
	return H
		
w = np.arange(0,5000,100)

plt.plot(w, (abs(H(1j*w))))
plt.xlabel('$\omega$')
plt.ylabel('$|H(e^{j\\omega})| $')
plt.grid()
plt.savefig('bt.png')
plt.show()
