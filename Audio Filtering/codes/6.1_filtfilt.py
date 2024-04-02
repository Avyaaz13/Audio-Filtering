import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def custom_lfilter(b, a, x):
    """
    Filter data along one dimension with an IIR or FIR filter.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence. If `a[0]` is not 1,
        then both `a` and `b` are normalized by `a[0]`.
    x : array_like
        An input array to be filtered.

    Returns
    -------
    y : array
        The output of the digital filter.
    """
    # Initialize the output array
    y = np.zeros_like(x)

    # Apply the filter
    M = len(b) - 1
    N = len(a) - 1
    for n in range(len(x)):
        for k in range(M+1):
            if n - k >= 0:
                y[n] += b[k] * x[n - k]
        for k in range(1, N+1):
            if n - k >= 0:
                y[n] -= a[k] * y[n - k]

    # Normalize if needed
    if a[0] != 1:
        y /= a[0]

    return y

def custom_filtfilt(b, a, x):
    # Perform forward filtering
    forward_output = custom_lfilter(b, a, x)

    # Reverse the signal
    reverse_output = custom_lfilter(b, a, np.flip(forward_output))

    # Reverse the reversed signal
    reverse_output = np.flip(reverse_output)

    return reverse_output

# Load the input audio
x, fs = sf.read('Input_audio.wav')

# Define the range of interest
range_of_interest = slice(1600, 2000)

# Order of the filter
order = 3   

# Cutoff frequency 4kHz
cutoff_freq = 4000.0  

# Digital frequency
Wn = 2 * cutoff_freq / fs 

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

# Filter the input signal with Butterworth filter using scipy's filtfilt
output_signal_scipy = signal.filtfilt(b, a, x)

# Filter the input signal with Butterworth filter using custom filtfilt
output_signal_custom = custom_filtfilt(b, a, x)

# Plot
plt.figure(figsize=(10, 6))
plt.stem(output_signal_scipy[range_of_interest], linefmt='c', markerfmt='go', basefmt='k', label='Scipy filtfilt')
plt.stem(output_signal_custom[range_of_interest], linefmt='b', markerfmt='ro', basefmt='k', label='Custom filtfilt')
plt.legend()
plt.grid(True)
plt.show()
plt.clf()

plt.stem(x[range_of_interest], linefmt='c', markerfmt='ro', basefmt='k', label='x(n):Input audio')
plt.xlabel('$n$')
plt.ylabel('$x(n)$')
plt.legend()
plt.grid(True)
plt.show()

