import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.optimize import fsolve

# Define parameters
beta = 0.5
N  = 5
t = np.linspace(0, 25, 10000)

# Function that reaches a non-zero steady state
f = gammainc(N, beta * t)

tau_i = (N - 1) / beta

def equation(tau, N, beta):
    return gammainc(N, beta * tau) - 0.5

tau_50 = fsolve(equation, x0=N/beta, args=(N, beta))[0]

tau_A = N / beta

# Plot results
plt.figure(figsize=(3.2, 2))
plt.plot(t, f)
plt.axvline(x = tau_i, color = 'goldenrod')
plt.text(tau_i - 0.9, -0.1, r'$\tau_i$', color = 'goldenrod')
plt.axhline(y = 0.5, color = 'c', linestyle = '--')
plt.axvline(x = tau_50, color = 'c')
plt.text(tau_50 - 1.1, -0.1, r'$\tau_{50}$', color = 'c')
plt.fill_between(t, f, 1, color = 'lavender', hatch = '\ ', edgecolor = 'mediumpurple', alpha = 0.5)
plt.axvline(x = tau_A, color = 'mediumpurple')
plt.text(tau_A + 0.2, -0.1, r'$\tau_A$', color = 'mediumpurple')
plt.xlabel(r'$t$')
plt.ylabel(r'$f(t)$')
plt.xticks([])
plt.yticks([])
plt.xlim(0, 25)
plt.ylim(0, 1.1)
plt.savefig("Steady_state.pdf")

# Show plot
plt.tight_layout()
plt.show()
