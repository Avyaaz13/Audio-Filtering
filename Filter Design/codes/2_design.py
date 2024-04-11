import numpy as np
import matplotlib.pyplot as plt

# Function to generate Chebyshev polynomial coefficients of order N
#def cheb(N):
#    if N == 0:
#        return np.array([1])
#    elif N == 1:
#        return np.array([1, 0])
#    else:
#        u = np.array([1])
#        v = np.array([1, 0])
#        for i in range(1, N):
#            p = np.convolve([2, 0], v)
#            m, n = len(p), len(u)
#            w = p + np.concatenate((np.zeros(m - n), u))
#            u = v
#            v = w
#        return w

# The low-pass Chebyshev design parameters
epsilon = 0.4
N = 4

# Analytically obtaining the roots of the Chebyshev polynomial
# in the left half of the complex plane
beta = ((np.sqrt(1 + epsilon ** 2) + 1) / epsilon) ** (1 / N)
r1 = (beta ** 2 - 1) / (2 * beta)
r2 = (beta ** 2 + 1) / (2 * beta)

# Obtaining the polynomial approximation for the low pass
# Chebyshev filter to obtain a stable filter
u = np.array([1])
for n in range(N // 2):
    phi = np.pi / 2 + (2 * n + 1) * np.pi / (2 * N)
    v = np.array([1, -2 * r1 * np.cos(phi), r1 ** 2 * np.cos(phi) ** 2 + r2 ** 2 * np.sin(phi) ** 2])
    p = np.convolve(v, u)
    u = p

# Evaluating the gain of the stable lowpass filter
# The gain has to be 1/sqrt(1+epsilon^2) at Omega = 1
G = np.abs(np.polyval(p, 1j))/ np.sqrt(1 + epsilon ** 2)

# Plotting the magnitude response of the stable filter and comparing with the desired response for the purpose
# of verification
Omega = np.arange(0, 2.01, 0.01)

H_stable = np.abs(G / np.polyval(p, 1j * Omega))

H = np.where(Omega < 1, 1 / np.sqrt(1 + epsilon**2 * (np.cos(N * np.arccos(Omega)))**2), 1 / np.sqrt(1 + epsilon**2 * (np.cosh(N * np.arccosh(Omega)))**2))

plt.plot(Omega, H_stable, 'bo',fillstyle = 'none', label='Design')
plt.plot(Omega, H, 'm', label='Specification')
plt.xlabel('$\\Omega$')
plt.ylabel('$|H_{a,LP}(j\\Omega)|$')
plt.legend()
plt.grid()
plt.savefig('fig2.png')
plt.show()
