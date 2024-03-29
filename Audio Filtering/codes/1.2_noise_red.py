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

# Filter the input signal with a Butterworth filter
output_signal = signal.filtfilt(b, a, input_signal, method="gust")

sf.write('ReducedNoise_s181.wav', output_signal, fs)

