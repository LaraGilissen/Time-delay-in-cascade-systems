import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma
from scipy.optimize import minimize

# Define ranges for N and beta
Ns = np.arange(1, 51) 
betas = np.arange(0.1, 10.175, 0.175)

# Define parameters
a = 0.1
r = 1

def equation(tau, N, beta, a, r):
    return -1/(beta-a)**N * r * np.exp(-a*tau)/gamma(N) * gammainc(N, (beta-a) * tau)

tau_m = np.zeros((len(betas), len(Ns)))

for i, beta in enumerate(betas):
    for j, N in enumerate(Ns):
        tau_m[i, j] = minimize(equation, x0=N/beta, args=(N, beta, a, r)).x[0]

N, beta = np.meshgrid(Ns, betas)

tau_g = N / beta + 1 / a


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
plt.savefig("beta_eq_exp.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_exp_beta.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5 ####################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, tau_m[28,:], label=r'$\tau_m$')
plt.plot(Ns, tau_g[28,:], label=r'$\tau_g$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("beta_eq_exp_N.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over N ##########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_m,1))
plt.fill_between(betas, np.mean(tau_m,1) - np.std(tau_m,1), np.mean(tau_m,1) + np.std(tau_m,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_m)$')

# Show plot
plt.savefig("beta_eq_exp_tm_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over beta #######################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_m,0))
plt.fill_between(Ns, np.mean(tau_m,0) - np.std(tau_m,0), np.mean(tau_m,0) + np.std(tau_m,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_m)$')

# Show plot
plt.savefig("beta_eq_exp_tm_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over N ##########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(betas, np.mean(tau_g,1))
plt.fill_between(betas, np.mean(tau_g,1) - np.std(tau_g,1), np.mean(tau_g,1) + np.std(tau_g,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Coefficients $\beta$')
plt.ylabel(r'$mean_N (\tau_g)$')

# Show plot
plt.savefig("beta_eq_exp_tg_beta_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over beta #######################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_g,0))
plt.fill_between(Ns, np.mean(tau_g,0) - np.std(tau_g,0), np.mean(tau_g,0) + np.std(tau_g,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{\beta} (\tau_g)$')

# Show plot
plt.savefig("beta_eq_exp_tg_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
