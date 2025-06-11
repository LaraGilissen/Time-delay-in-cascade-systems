import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(12345)

# Define parameters
alpha = 1 # product of all alphas
r = 1
t0 = 5
a = 0.1
t = np.linspace(0, 50, 1000)
w = 1

# Calculate f_N
def G_N(u, alpha, betas):
    N = len(betas)
    total = 0.0
    for k in range(N):
        prod = 1
        for l in range(N):
            if l != k:
                prod *= (betas[l] - betas[k])
        total += np.exp(-betas[k] * u) / prod
    return alpha * total

# Calculate output for constant input
def constant(t, alpha, betas, r):
    constant_output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        integral, _ = quad(G_N, 0, t_val, args=(alpha, betas), epsabs=1e-15, epsrel=1e-15)
        constant_output[i] = r * integral
    return constant_output

# Calculate output for pulse input
def pulse(t, alpha, betas, r, t0):
    pulse_output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        # Determine integration limits based on t and t_0
        if t_val <= t0:
            a, b = 0, t_val
        else:
            a, b = t_val - t0, t_val
        integral, _ = quad(G_N, a, b, args=(alpha, betas), epsabs=1e-15, epsrel=1e-15)
        pulse_output[i] = r * integral
    return pulse_output

# Calculate output for exponential input
def exponential(t, alpha, betas, r, a):
    exp_output = np.zeros_like(t)
    def integrand(u, t, alpha, betas):
        return G_N(u, alpha, betas) * np.exp(-a * (t - u))
    for i, t_val in enumerate(t):
        integral, _ = quad(integrand, 0, t_val, args=(t_val, alpha, betas), epsabs=1e-15, epsrel=1e-15)
        exp_output[i] = r * integral
    return exp_output


#########
# Plots #
#########

# c = 0.5, N = 5 ##############################################################################################
c = 0.5
N = 5
betas = np.random.uniform(c - w/2, c + w/2, size=N)
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, betas, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, betas, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, betas, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_diff_output_c0.5_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# c = 0.5, N = 10 #############################################################################################
c = 0.5
N = 10
betas = np.random.uniform(c - w/2, c + w/2, size=N)
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, betas, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, betas, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, betas, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_diff_output_c0.5_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# c = 5, N = 5 ################################################################################################
c = 5
N = 5
betas = np.random.uniform(c - w/2, c + w/2, size=N)
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, betas, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, betas, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, betas, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_diff_output_c5_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# c = 5, N = 10 ###############################################################################################
c = 5
N = 10
betas = np.random.uniform(c - w/2, c + w/2, size=N)
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, betas, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, betas, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, betas, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_diff_output_c5_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()