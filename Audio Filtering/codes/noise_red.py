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

# b and a are numerator and denominator polynomials, respectively
b, a = signal.butter(order, Wn, 'low')

# Ensure the signal is long enough for the filter
if len(input_signal) < max(3 * (max(len(a), len(b)) - 1), 15):
    raise ValueError("Input signal is too short for the specified filter order and padding.")

print(a)
print(b)

# Filter the input signal with a Butterworth filter
output_signal = signal.filtfilt(b, a, input_signal, method="gust")

#output_signal = signal.lfilter(b,a, input_signal)

sf.write('ReducedNoise_s181.wav', output_signal, fs)

