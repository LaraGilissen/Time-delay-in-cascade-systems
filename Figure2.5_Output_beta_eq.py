import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma, gammainc

# Define parameters
alpha = 1 # product of all alphas
r = 1
t0 = 5
a = 0.1
t = np.linspace(0, 50, 10000)

# Calculate output for constant input
def constant(t, alpha, beta, N, r):
    constant_output = alpha/beta**N * r * gammainc(N, beta * t)
    return constant_output

# Calculate output for pulse input
def pulse(t, alpha, beta, N, r, t0):
    pulse_output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if t_val <= t0:
            pulse_output[i] = alpha/beta**N * r * gammainc(N, beta * t_val)
        else:
            pulse_output[i] = alpha/beta**N * r * (gammainc(N, beta * t_val) - gammainc(N, beta * (t_val - t0)))
    return pulse_output

# Calculate output for exponential input
def exponential(t, alpha, beta, N, r, a):
    exp_output = np.zeros_like(t)
    # Integrand for the convolution
    def integrand(u, t):
        return beta**N * u**(N-1) * np.exp(-beta*u) / gamma(N) * np.exp(-a * (t-u))
    
    # Perform numerical integration
    for i, t_val in enumerate(t):
        integral, _ = quad(integrand, 0, t_val, args=(t_val,))
        exp_output[i] = alpha/beta**N * r * integral
    return exp_output


#########
# Plots #
#########

# beta = 0.5, N = 5 ###########################################################################################
beta = 0.5
N = 5
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_eq_output_beta0.5_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 0.5, N = 10 ##########################################################################################
beta = 0.5
N = 10
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_eq_output_beta0.5_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5, N = 5 #############################################################################################
beta = 5
N = 5
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_eq_output_beta5_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5, N = 10 ############################################################################################
beta = 5
N = 10
plt.figure(figsize=(3.1, 2))
plt.plot(t, constant(t, alpha, beta, N, r), label=r'Constant $R$')
plt.plot(t, pulse(t, alpha, beta, N, r, t0), label=r'Pulse $R$')
plt.plot(t, exponential(t, alpha, beta, N, r, a), label=r'Exponential $R$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x_N(t)$')
plt.legend()

# Show plot
plt.savefig("beta_eq_output_beta5_N10.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()