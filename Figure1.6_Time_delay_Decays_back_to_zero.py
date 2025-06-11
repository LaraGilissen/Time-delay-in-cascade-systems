import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc

# Define parameters
beta = 0.5
N  = 5
t0 = 5
t = np.linspace(0, 35, 10000)

# Function that increases and decays back to zero
f = np.zeros_like(t)
for i, t_val in enumerate(t):
    if t_val <= t0:
        f[i] = gammainc(N, beta * t_val)
    else:
        f[i] = (gammainc(N, beta * t_val) - gammainc(N, beta * (t_val - t0)))

tau_m = t0 / (1 - np.exp(-beta * t0 / (N - 1)))

tau_g = (t0 / 2) + (N / beta)

# Plot results
plt.figure(figsize=(3.1, 2))
plt.plot(t, f)
plt.axvline(x = tau_m, color = 'goldenrod')
plt.text(tau_m - 1, -0.05, r'$\tau_m$', color = 'goldenrod')
plt.axvline(x = tau_g, color = 'mediumpurple')
plt.text(tau_g - 0.3, -0.05, r'$\tau_g$', color = 'mediumpurple')
plt.xlabel(r'$t$')
plt.ylabel(r'$f(t)$')
plt.xticks([])
plt.yticks([])
plt.xlim(0, 35)
plt.ylim(0, 0.5)
plt.savefig("Decays_back_to_zero.pdf")

# Show plot
plt.tight_layout()
plt.show()
