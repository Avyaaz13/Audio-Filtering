import numpy as np
import matplotlib.pyplot as plt

# Define the filter coefficients
a = [1, -1.87302725, 1.30032695, -0.31450204]
b = [0.0140997, 0.0422991, 0.0422991, 0.0140997]

# Compute the roots (poles and zeros)
poles = np.roots(a)
zeros = np.roots(b)

print("Poles:", poles)
print("Zeros:", zeros)


# Plot the pole-zero plot
plt.scatter(poles.real, poles.imag, marker='x', color='r', label='Poles')
plt.scatter(zeros.real, zeros.imag, marker='o', color='b', label='Zeros')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Real')
plt.ylabel('Imaginary')
#plt.title('Pole-Zero Plot')
plt.legend()
plt.ylim(-6, 6)
plt.grid(True)
plt.savefig('pole_zero.png')
plt.show()

