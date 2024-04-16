import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
colors = ['b', 'g', 'r', 'm', 'c', 'purple', 'y']
for N in range(4, 5):  # N = 4
    for i, epsilon in enumerate(np.arange(0.35, 0.64, 0.05)):
        Omega = np.arange(0, 3.02, 0.02)
        H = np.where(Omega < 1, 1 / np.sqrt(1 + epsilon**2 * (np.cos(N * np.arccos(Omega)))**2), 1 / np.sqrt(1 + epsilon**2 * (np.cosh(N * np.arccosh(Omega)))**2))
        ax.plot(Omega, H, color=colors[i], label=f'$\\epsilon = {epsilon:.2f}$')

passband = (Omega >= 0) & (Omega <= 1)
transitionband = (Omega >= 1) & (Omega <= 1.459)
stopband = (Omega >= 1.459) & (Omega <= 3)
ax.fill_between(Omega, 0, 1, where=passband, color='cyan', alpha=0.4, label='Passband')
ax.fill_between(Omega, 0, 1, where=transitionband, color='yellow', alpha=0.5, label='Transitionband')
ax.fill_between(Omega, 0, 1, where=stopband, color='#00FF00', alpha=0.8, label='Stopband')

plt.grid()
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$|H_{a,LP}(j\Omega)|$')
# plt.text(1.25, 0.45, r'$\epsilon = 0.35$')
# plt.text(1.0, 0.3, r'$\epsilon = 0.6$')
plt.legend()
plt.savefig('fig1.png')
plt.show()
