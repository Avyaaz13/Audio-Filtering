import numpy as np
import matplotlib.pyplot as plt

def H(z):
	num = np.polyval([1,0,1],z**(-1))
	den = np.polyval([0.5,1],z**(-1))
	H = num/den
	return H		

omega = np.linspace(-3*np.pi,3*np.pi,100)

plt.plot(omega, abs(H(np.exp(1j*omega))))
plt.xlabel('$\omega$')
plt.ylabel('$|H(e^{\jmath\omega})| $')
plt.grid()

plt.savefig("H(z).png")
