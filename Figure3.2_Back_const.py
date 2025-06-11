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
    return G_vals

def find_tau_i(t_vals, G_vals):
    result = t_vals[np.argmax(G_vals)]
    return result

def find_tau_50(t_vals, G_vals):
    x_N_vals = r * cumulative_trapezoid(G_vals, t_vals, initial=0)
    result = t_vals[np.argmin(np.abs(x_N_vals-1/2*x_N_vals[-1]))]
    return result

def find_tau_A(alpha, beta, gamma, N):
    s = mpf('0.001')
    G_hat = lambda s: laplace_G(s, alpha, beta, gamma, N)
    dG_hat_ds = diff(G_hat, s)
    tau_A = -dG_hat_ds / G_hat(s)
    return float(tau_A.real)

tau_i = np.zeros((len(betas), len(Ns)))
tau_50 = np.zeros((len(betas), len(Ns)))
tau_A = np.zeros((len(betas), len(Ns)))

for i, beta in enumerate(betas):
    for j, N in enumerate(Ns):
        t_vals = np.linspace(0.001, 100, 1000)
        G_vals = G(alpha, beta, gamma, N, t_vals)
        tau_i[i, j] = find_tau_i(t_vals, G_vals)
        tau_50[i, j] = find_tau_50(t_vals, G_vals)
        tau_A[i, j] = find_tau_A(alpha, beta, gamma, N)


#########
# Plots #
#########

N, beta = np.meshgrid(Ns, betas)

# Tau_i, tau_50 and tau_A #####################################################################################
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.3), sharey=True, constrained_layout=True)
c_i = ax[0].pcolor(N, beta, np.log10(tau_i), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)
c_50 = ax[1].pcolor(N, beta, np.log10(tau_50), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)
c_A = ax[2].pcolor(N, beta, np.log10(tau_A), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax[0].tick_params(axis='both', which='major')
ax[0].set_xlabel(r'Number of states $N$')
ax[0].set_ylabel(r'Coefficients $\beta$')
ax[1].tick_params(axis='both', which='major')
ax[1].set_xlabel(r'Number of states $N$')
ax[1].set_ylabel(r'Coefficients $\beta$')
ax[2].tick_params(axis='both', which='major')
ax[2].set_xlabel(r'Number of states $N$')
ax[2].set_ylabel(r'Coefficients $\beta$')

# Colorbar
cbar = fig.colorbar(c_i, ax=ax.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.03)
cbar.set_label(r'log$_{10}(\tau)$')
cbar.ax.tick_params()

# Save and show plot
plt.savefig("back_const.pdf", bbox_inches = 'tight')
plt.show()


# N = 10 ######################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, tau_i[:,9], label=r'$\tau_i$')
plt.plot(betas, tau_50[:,9], label=r'$\tau_{50}$')
plt.plot(betas, tau_A[:,9], label=r'$\tau_A$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("back_const_beta.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5 ####################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, tau_i[10,:], label=r'$\tau_i$')
plt.plot(Ns, tau_50[10,:], label=r'$\tau_{50}$')
plt.plot(Ns, tau_A[10,:], label=r'$\tau_A$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("back_const_N.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_i over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_i,1))
plt.fill_between(betas, np.mean(tau_i,1) - np.std(tau_i,1), np.mean(tau_i,1) + np.std(tau_i,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_i)$')

# Show plot
plt.savefig("back_const_ti_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_i over beta ########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_i,0))
plt.fill_between(Ns, np.mean(tau_i,0) - np.std(tau_i,0), np.mean(tau_i,0) + np.std(tau_i,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_i)$')

# Show plot
plt.savefig("back_const_ti_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_50 over N ##########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_50,1))
plt.fill_between(betas, np.mean(tau_50,1) - np.std(tau_50,1), np.mean(tau_50,1) + np.std(tau_50,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_{50})$')

# Show plot
plt.savefig("back_const_t50_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_50 over beta #######################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_50,0))
plt.fill_between(Ns, np.mean(tau_50,0) - np.std(tau_50,0), np.mean(tau_50,0) + np.std(tau_50,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_{50})$')

# Show plot
plt.savefig("back_const_t50_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_A over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_A,1))
plt.fill_between(betas, np.mean(tau_A,1) - np.std(tau_A,1), np.mean(tau_A,1) + np.std(tau_A,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_A)$')

# Show plot
plt.savefig("back_const_tA_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_A over beta ########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_A,0))
plt.fill_between(Ns, np.mean(tau_A,0) - np.std(tau_A,0), np.mean(tau_A,0) + np.std(tau_A,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_A)$')

# Show plot
plt.savefig("back_const_tA_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
