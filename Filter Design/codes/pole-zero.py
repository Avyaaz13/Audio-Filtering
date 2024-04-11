import numpy as np
import matplotlib.pyplot as plt

# Given parameters
epsilon = 0.4
N = 4
Omega = np.arange(0, 1, 0.1)
# Calculate beta
beta = ((np.sqrt(1 + epsilon ** 2) + 1) / epsilon) ** (1 / N)
H =np.sqrt(1 + epsilon**2 *((8*(Omega**4) - 8*(Omega**2) + 1))**2)
# Compute r1 and r2
r1 = (beta ** 2 - 1) / (2 * beta)
r2 = (beta ** 2 + 1) / (2 * beta)

# Compute poles
poles = []
for k in range(2*N):
    phi_k = np.pi / 2 + (2 * k + 1) * np.pi / (2 * N)
    pole = r1 * np.cos(phi_k) + 1j * r2 * np.sin(phi_k)
    poles.append(pole)
print(poles)

plt.scatter(np.real(poles), np.imag(poles), color='green', marker='x')
plt.title('Pole-Zero Plot')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.xlim(-2,2)
plt.ylim(-3,3)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('pole_zero.png')
plt.show()

