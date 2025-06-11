import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
from scipy.integrate import cumulative_trapezoid

# Define ranges for N and beta
Ns = np.arange(1, 51) 
betas = np.arange(3.001, 10.199, 0.199)

# Define parameters
alpha = 1
gamma = 0.25
r = 1
a = 0.1

# Set mpmath precision
mp.dps = 5

def laplace_G(s, alpha, beta, gamma, N):
    s = mpmathify(s)
    sqrt_term = sqrt((s + beta)**2 - 4 * alpha * gamma)
    lambda_plus = (s + beta + sqrt_term) / 2
    lambda_minus = (s + beta - sqrt_term) / 2
    numerator = alpha**N * (lambda_plus - lambda_minus)
    denominator = lambda_plus**(N+1) - lambda_minus**(N+1)
    return numerator / denominator

def G(alpha, beta, gamma, N, t_vals):
    laplace = lambda s: laplace_G(s, alpha, beta, gamma, N)
    G_vals = np.array([float(invertlaplace(laplace, t, method='talbot')) for t in t_vals])
    G_vals = np.maximum(G_vals,0)
    return G_vals

def find_tau_m(t_vals, G_vals, a, r):
    x_N_vals = np.zeros_like(t_vals)
    for k, t in enumerate(t_vals):
        integrand = G_vals[:k+1] * np.exp(-a * (t - t_vals[:k+1]))
        x_N_vals[k] = r * cumulative_trapezoid(integrand, t_vals[:k+1], initial=0)[-1]
    result = t_vals[np.argmax(x_N_vals)]
    return result

def find_tau_g(alpha, beta, gamma, N):
    G = lambda s: laplace_G(s, alpha, beta, gamma, N)
    numerator = diff(G,0)
    denominator = G(0)
    result = max(1/a -  float(re(numerator/denominator)), 1e-6)
    return result

tau_m = np.zeros((len(betas), len(Ns)))
tau_g = np.zeros((len(betas), len(Ns)))

for i, beta in enumerate(betas):
    for j, N in enumerate(Ns):
        t_vals = np.linspace(0.001, 100, 1000)
        G_vals = G(alpha, beta, gamma, N, t_vals)
        tau_m[i, j] = find_tau_m(t_vals, G_vals, a, r)
        tau_g[i, j] = find_tau_g(alpha, beta, gamma, N)


#########
# Plots #
#########

N, beta = np.meshgrid(Ns, betas)

# Tau_m and tau_g #############################################################################################
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True, constrained_layout=True)
c_m = ax[0].pcolor(N, beta, np.log10(tau_m), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)
c_g = ax[1].pcolor(N, beta, np.log10(tau_g), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax[0].tick_params(axis='both', which='major')
ax[0].set_xlabel(r'Number of states $N$')
ax[0].set_ylabel(r'Coefficients $\beta$')
ax[1].tick_params(axis='both', which='major')
ax[1].set_xlabel(r'Number of states $N$')
ax[1].set_ylabel(r'Coefficients $\beta$')

# Colorbar
cbar = fig.colorbar(c_m, ax=ax.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.03)
cbar.set_label(r'log$_{10}(\tau)$')
cbar.ax.tick_params()

# Save and show plot
plt.savefig("back_exp.pdf", bbox_inches = 'tight')
plt.show()


# N = 10 ######################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, tau_m[:,9], label=r'$\tau_m$')
plt.plot(betas, tau_g[:,9], label=r'$\tau_g$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("back_exp_beta.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5 ####################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, tau_m[10,:], label=r'$\tau_m$')
plt.plot(Ns, tau_g[10,:], label=r'$\tau_g$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("back_exp_N.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_m,1))
plt.fill_between(betas, np.mean(tau_m,1) - np.std(tau_m,1), np.mean(tau_m,1) + np.std(tau_m,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_m)$')

# Show plot
plt.savefig("back_exp_tm_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over beta ########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_m,0))
plt.fill_between(Ns, np.mean(tau_m,0) - np.std(tau_m,0), np.mean(tau_m,0) + np.std(tau_m,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_m)$')

# Show plot
plt.savefig("back_exp_tm_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_g,1))
plt.fill_between(betas, np.mean(tau_g,1) - np.std(tau_g,1), np.mean(tau_g,1) + np.std(tau_g,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_g)$')

# Show plot
plt.savefig("back_exp_tg_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over beta ########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_g,0))
plt.fill_between(Ns, np.mean(tau_g,0) - np.std(tau_g,0), np.mean(tau_g,0) + np.std(tau_g,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_g)$')

# Show plot
plt.savefig("back_exp_tg_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
