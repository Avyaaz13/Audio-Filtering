import numpy as np
import matplotlib.pyplot as plt

# Filter number
L = 1

# Sampling frequency (kHz)
Fs = 48

# Constant used to get the normalized digital frequencies
const = 2 * np.pi / Fs

# The permissible filter amplitude deviation from unity
delta = 0.15

# Bandpass filter specifications (kHz)
F_p1 = 4 + 0.6 * (L + 2)
F_p2 = 4 + 0.6 * L

# Transition band
delF = 0.3

# Stopband
F_s1 = F_p1 + 0.3
F_s2 = F_p2 - 0.3

# Normalized digital filter specifications (radians/sec)
omega_p1 = const * F_p1
omega_p2 = const * F_p2

omega_c = (omega_p1 + omega_p2) / 2
omega_l = (omega_p1 - omega_p2) / 2

omega_s1 = const * F_s1
omega_s2 = const * F_s2
delomega = 2 * np.pi * delF / Fs

N = 100  # Adjust N if needed
n = np.arange(-N, N + 1)
hbp = 2*np.sin(n * omega_l)*np.cos(13*n*np.pi/60) / np.where(n != 0,(n * np.pi),1)
hbp[N] = 1/ 40

# The lowpass filter plot
#omega = np.linspace(-np.pi / 2, np.pi / 2, 400)
plt.stem(n, hbp,linefmt='b',markerfmt='ro',basefmt = 'k')
plt.xlabel(r'$n$')
plt.ylabel(r'$h_{bp}(n)$')
plt.grid()
plt.savefig('hbp.png')
plt.show()

