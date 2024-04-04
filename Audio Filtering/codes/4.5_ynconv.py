import numpy as np
import matplotlib.pyplot as plt


n = np.arange(14)
fn=(-1/2)**n
hn1=np.pad(fn, (0,2), 'constant', constant_values=(0))
hn2=np.pad(fn, (2,0), 'constant', constant_values=(0))
h = hn1+hn2

nh=len(h)
x=np.array([1.0,2.0,3.0,4.0,2.0,1.0])
nx = len(x)

y = np.zeros(nx+nh-3)

for n in range(0,nx+nh-3):
	for k in range(0,nx):
		if n-k >= 0 and n-k < nh:
			y[n]+=x[k]*h[n-k]


plt.stem(range(0,nx+nh-3),y,linefmt='b-', markerfmt='ro', basefmt='k')
plt.xlabel('$n$')
plt.ylabel('$y(n)$')
plt.grid()

plt.savefig('y_by_conv.png')

