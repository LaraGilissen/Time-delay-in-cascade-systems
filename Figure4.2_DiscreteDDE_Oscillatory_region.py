import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define range for m
m = np.linspace(1, 30, 1000)

# Define parameters
ratios = range(1, 11)
alpha = 1
K = 0.5

def find_steady_state(x, K, m, alpha, beta):
    return K**m/(K**m + x**m) - beta/alpha * x

plt.figure(figsize=(6.2, 4))

for ratio in ratios:
    beta = alpha/(ratio*K)
    steady_state = np.zeros_like(m)
    for i, mval in enumerate(m):
        steady_state[i] = fsolve(lambda x: find_steady_state(x,K,mval,alpha,beta), K)

    omegasq = (alpha * m * K**m * steady_state**(m-1) / (K**m + steady_state**m)**2)**2 - beta**2

    omega = np.full_like(omegasq, np.nan)
    valid = omegasq > 0
    omega[valid] = np.sqrt(omegasq[valid])

    N = beta/omega * (np.arctan(-omega/beta) + np.pi) + 1
    
    plt.plot(m, N, label=fr'$\frac{{\alpha_1}}{{\beta K}}$={ratio}', color='tab:blue', alpha=ratio/10)

N_inf = (np.arctan(-np.sqrt(m**2 - 1)) + np.pi)/np.sqrt(m**2 - 1) + 1

plt.plot(m, N_inf, label=fr'$\frac{{\alpha_1}}{{\beta K}} \rightarrow \infty$', color='midnightblue')
plt.xlabel(r'Hill exponent $m$')
plt.ylabel(r'Number of states $N$')
plt.xlim(0, 30)
plt.ylim(0, 10)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Closed_loop_discrete.pdf", bbox_inches = 'tight')
plt.show()