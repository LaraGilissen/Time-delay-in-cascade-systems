import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.optimize import fsolve

# Define ranges for N and beta
Ns = np.arange(1, 51) 
betas = np.arange(0.1, 10.175, 0.175)

# Define parameter
r = 1

def equation(tau, N, beta, lim_val):
    return (1 / beta**N) * r * gammainc(N, beta * tau) - (lim_val / 2)

tau_50 = np.zeros((len(betas), len(Ns)))

for i, beta in enumerate(betas):
    for j, N in enumerate(Ns):
        lim_val = (1 / beta**N) * r * gammainc(N, beta * 1e10)
        tau_50[i, j] = fsolve(equation, x0=N/beta, args=(N, beta, lim_val))[0]

N, beta = np.meshgrid(Ns, betas)

tau_i = (N - 1) / beta

tau_A = N / beta


#########
# Plots #
#########

# Tau_i #######################################################################################################
fig, ax = plt.subplots(figsize=(3.1, 2))
c_i = ax.pcolor(N, beta, np.log10(tau_i), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax.tick_params(axis='both', which='major')
ax.set_xlabel(r'Number of states $N$')
ax.set_ylabel(r'Coefficients $\beta$')

# Colorbar
cbar_i = fig.colorbar(c_i)
cbar_i.set_label(r'log$_{10}(\tau_i)$')
cbar_i.ax.tick_params()

# Save and show plot
plt.savefig("beta_eq_const_ti.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Tau_50 ######################################################################################################
fig, ax = plt.subplots(figsize=(3.1, 2))
c_50 = ax.pcolor(N, beta, np.log10(tau_50), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax.tick_params(axis='both', which='major')
ax.set_xlabel(r'Number of states $N$')
ax.set_ylabel(r'Coefficients $\beta$')

# Colorbar
cbar_50 = fig.colorbar(c_50)
cbar_50.set_label(r'log$_{10}(\tau_{50})$')
cbar_50.ax.tick_params()

# Save and show plot
plt.savefig("beta_eq_const_t50.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Tau_A #######################################################################################################
fig, ax = plt.subplots(figsize=(3.1, 2))
c_A = ax.pcolor(N, beta, np.log10(tau_A), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax.tick_params(axis='both', which='major')
ax.set_xlabel(r'Number of states $N$')
ax.set_ylabel(r'Coefficients $\beta$')

# Colorbar
cbar_A = fig.colorbar(c_A)
cbar_A.set_label(r'log$_{10}(\tau_A)$')
cbar_A.ax.tick_params()

# Save and show plot
plt.savefig("beta_eq_const_tA.pdf", bbox_inches = 'tight')
plt.tight_layout()
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
plt.savefig("beta_eq_const_beta.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# beta = 5 ####################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, tau_i[28,:], label=r'$\tau_i$')
plt.plot(Ns, tau_50[28,:], label=r'$\tau_{50}$')
plt.plot(Ns, tau_A[28,:], label=r'$\tau_A$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("beta_eq_const_N.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_ti_beta_mean.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_ti_N_mean.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_t50_beta_mean.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_t50_N_mean.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_tA_beta_mean.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_eq_const_tA_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
