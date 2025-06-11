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

def find_N(N, x, K, m, alpha, beta):
    return (np.cos(np.pi/N))**N - beta * (K**m+x**m)**2 / (alpha*m*K**m*x**(m-1))

def find_N_inf(N, m):
    return (np.cos(np.pi/N))**N - 1/m

plt.figure(figsize=(6.2, 4))

for ratio in ratios:
    beta = alpha/(ratio*K)
    steady_state = np.zeros_like(m)
    N = np.zeros_like(m)
    N_inf = np.zeros_like(m)
    for i, mval in enumerate(m):
        steady_state[i] = fsolve(lambda x: find_steady_state(x,K,mval,alpha,beta), K)
        Nvals = np.linspace(2, 100, 10000)
        N[i] = Nvals[np.argmin(abs(find_N(Nvals, steady_state[i], K, mval, alpha, beta)))]
        N_inf[i] = Nvals[np.argmin(abs(find_N_inf(Nvals, mval)))]

    plt.plot(m, N, label=fr'$\frac{{\alpha_1}}{{\beta K}}$={ratio}', color='tab:blue', alpha=ratio/10)

N2 = (np.arctan(-np.sqrt(m**2 - 1)) + np.pi)/np.sqrt(m**2 - 1) + 1

plt.plot(m, N_inf, label=fr'$\frac{{\alpha_1}}{{\beta K}} \rightarrow \infty$', color='midnightblue')
plt.xlabel(r'Hill exponent $m$')
plt.ylabel(r'Number of states $N$')
plt.xlim(0, 30)
plt.ylim(0, 10)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Closed_loop_distributed.pdf", bbox_inches = 'tight')
plt.show()