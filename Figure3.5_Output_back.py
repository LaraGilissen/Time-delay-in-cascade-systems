import numpy as np
import matplotlib.pyplot as plt
from mpmath import invertlaplace, mp, sqrt
from scipy.integrate import cumulative_trapezoid

# Parameters
alpha = 1
gamma = 0.25
r = 1
t0 = 5
a = 0.1
t = np.linspace(0.001, 50, 1000)

# Set precision
mp.dps = 5

# Define Ĝ(s)
def Laplace_G(s, alpha, beta, gamma, N):
    Delta = (s + beta)**2 - 4 * alpha * gamma
    lam_p = (s + beta + sqrt(Delta)) / 2
    lam_m = (s + beta - sqrt(Delta)) / 2
    numerator = lam_p - lam_m
    denominator = lam_p**(N + 1) - lam_m**(N + 1)
    return alpha**N * numerator / denominator

# Compute G(t) numerically
def G(t, alpha, beta, gamma, N):
    G_hat = lambda s: Laplace_G(s, alpha, beta, gamma, N)
    G_vals = np.array([float(invertlaplace(G_hat, t_val, method='talbot')) for t_val in t])
    return G_vals

# Calculate output for constant input
def constant(t, alpha, beta, gamma, N, r):
    G_vals = G(t, alpha, beta, gamma, N)
    constant_output = r * cumulative_trapezoid(G_vals, t, initial=0)
    return constant_output

# Calculate output for pulse input
def pulse(t, alpha, beta, gamma, N, r, t0):
    G_vals = G(t, alpha, beta, gamma, N)
    pulse_output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if t_val <= t0:
            # Case 1: t <= t0, integrate from 0 to t
            pulse_output[i] = r * cumulative_trapezoid(G_vals[:i+1], t[:i+1], initial=0)[-1]
        else:
            # Case 2: t > t0, integrate from t - t0 to t
            idx_start = np.searchsorted(t, t_val - t0)  # Find the index corresponding to t - t0
            pulse_output[i] = r * cumulative_trapezoid(G_vals[idx_start:i+1], t[idx_start:i+1], initial=0)[-1]
    return pulse_output

# Calculate output for exponential input
def exponential(t, alpha, beta, gamma, N, r, a):
    G_vals = G(t, alpha, beta, gamma, N)
    exp_output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        weighted_G_vals = G_vals[:i+1] * np.exp(-a * (t_val - t[:i+1]))
        exp_output[i] = r * cumulative_trapezoid(weighted_G_vals, t[:i+1], initial=0)[-1]
    return exp_output


#########
# Plots #
#########

# beta = 3, N = 5 #############################################################################################
beta = 3
N = 5
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, gamma, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, gamma, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, gamma, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("back_output_beta3_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 3, N = 10 ############################################################################################
beta = 3
N = 10
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, gamma, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, gamma, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, gamma, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("back_output_beta3_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5, N = 5 #############################################################################################
beta = 5
N = 5
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, gamma, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, gamma, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, gamma, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("back_output_beta5_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5, N = 10 ############################################################################################
beta = 5
N = 10
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, gamma, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, gamma, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, gamma, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("back_output_beta5_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()