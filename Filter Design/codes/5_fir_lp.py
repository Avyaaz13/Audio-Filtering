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

# The Kaiser window design
A = -20 * np.log10(delta)
N = np.ceil((A - 8) / (4.57 * delomega))
N = 100  # Adjust N if needed
n = np.arange(-N, N + 1)
hlp = np.sin(n * omega_l) / (n * np.pi)
hlp[N] = omega_l / np.pi

# The Bandpass filter
hbp = 2 * hlp * np.cos(n * omega_c)

# The lowpass filter plot
omega = np.linspace(-*np.pi/2 , np.pi / 2, 400)
Hlp = np.abs(np.polyval(hlp, np.exp(-1j * omega)))
plt.plot(omega / np.pi, Hlp,'b')
plt.xlabel(r'$\omega/\pi$')
plt.ylabel(r'$|H_{lp}(\omega)|$')
plt.grid()
plt.savefig('fig5.png')
plt.show()

# fir_coeff = hbp

# save fir_coeff.dat fir_coeff -ascii
