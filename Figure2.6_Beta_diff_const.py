import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from scipy.optimize import fsolve

np.random.seed(12345)

# Define ranges for N and c, width and number of runs
Ns = np.arange(1, 51) 
cs = np.arange(0.5, 10.175, 0.18)
w = 1.0
runs = 50

def GNprime(t, betas):
    N = len(betas)
    total = 0.0
    for k in range(N):
        logprod = 0
        for l in range(N):
            if l != k:
                logprod += np.log(betas[l] - betas[k])
        total += betas[k] * np.exp(-betas[k] * t) / np.exp(logprod)
    return total

def find_tau_i(betas, u0):
    result = fsolve(GNprime, x0=u0, args=(betas))[0]
    return result

def equation(tau, betas):
    N = len(betas)
    total = 0
    for k in range(N):
        prod = 1
        for l in range(N):
            if l != k:
                prod *= betas[l] / (betas[l] - betas[k])
        total += np.exp(-betas[k] * tau) * prod
    return total - 0.5

def find_tau_50(betas, u0):
    result = fsolve(equation, x0=u0, args=(betas))[0]
    return result

def find_tau_A(betas):
    N = len(betas)
    result = 0
    for k in range(N):
        result += 1 / betas[k]
    return result

tau_i = np.zeros((len(cs), len(Ns)))
tau_50 = np.zeros((len(cs), len(Ns)))
tau_A = np.zeros((len(cs), len(Ns)))

for i, c in enumerate(cs):
    for j, N in enumerate(Ns):
        tau_is = []
        tau_50s = []
        tau_As = []
        for _ in range(runs):
            # Generate N beta values from U(c-w/2, c+w/2)
            betas = np.random.uniform(c - w/2, c + w/2, size=N)
            u0 = N/c
            if N == 1: # No inflection point when N = 1
                tau_i_result = np.nan
            else:
                tau_i_result = find_tau_i(betas, u0)
            tau_is.append(tau_i_result)
            tau_50_result = find_tau_50(betas, u0)
            tau_50s.append(tau_50_result)
            tau_A_result = find_tau_A(betas)
            tau_As.append(tau_A_result)
        tau_i[i, j] = trim_mean(tau_is, 0.05)
        tau_50[i, j] = trim_mean(tau_50s, 0.05)
        tau_A[i, j] = trim_mean(tau_As, 0.05)


#########
# Plots #
#########

N, c = np.meshgrid(Ns, cs)

# Tau_i, tau_50 and tau_A #####################################################################################
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True, constrained_layout=True)
c_i = ax[0].pcolor(N, c, np.log10(tau_i), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)
c_A = ax[1].pcolor(N, c, np.log10(tau_A), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax[0].tick_params(axis='both', which='major')
ax[0].set_xlabel(r'Number of states $N$')
ax[0].set_ylabel('Centre of interval \n coefficients $c$')
ax[1].tick_params(axis='both', which='major')
ax[1].set_xlabel(r'Number of states $N$')
ax[1].set_ylabel('Centre of interval \n coefficients $c$')

# Colorbar
cbar = fig.colorbar(c_i, ax=ax.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.03)
cbar.set_label(r'log$_{10}(\tau)$')
cbar.ax.tick_params()

# Save and show plot
plt.savefig("beta_diff_const.pdf", bbox_inches = 'tight')
plt.show()


# N = 10 ######################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, tau_i[:,9], label=r'$\tau_i$')
plt.plot(cs, tau_50[:,9], label=r'$\tau_{50}$')
plt.plot(cs, tau_A[:,9], label=r'$\tau_A$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("beta_diff_const_c.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_diff_const_N.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_i over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, np.nanmean(tau_i,1))
plt.fill_between(cs, np.nanmean(tau_i,1) - np.nanstd(tau_i,1), np.nanmean(tau_i,1) + np.nanstd(tau_i,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'$mean_N (\tau_i)$')

# Show plot
plt.savefig("beta_diff_const_ti_c_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_i over c ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_i,0))
plt.fill_between(Ns, np.mean(tau_i,0) - np.std(tau_i,0), np.mean(tau_i,0) + np.std(tau_i,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{c} (\tau_i)$')

# Show plot
plt.savefig("beta_diff_const_ti_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_50 over N ##########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, np.mean(tau_50,1))
plt.fill_between(cs, np.mean(tau_50,1) - np.std(tau_50,1), np.mean(tau_50,1) + np.std(tau_50,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'$mean_N (\tau_{50})$')

# Show plot
plt.savefig("beta_diff_const_t50_c_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_50 over c ##########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_50,0))
plt.fill_between(Ns, np.mean(tau_50,0) - np.std(tau_50,0), np.mean(tau_50,0) + np.std(tau_50,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{c} (\tau_{50})$')

# Show plot
plt.savefig("beta_diff_const_t50_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_A over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, np.mean(tau_A,1))
plt.fill_between(cs, np.mean(tau_A,1) - np.std(tau_A,1), np.mean(tau_A,1) + np.std(tau_A,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'$mean_N (\tau_A)$')

# Show plot
plt.savefig("beta_diff_const_tA_c_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_A over c ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_A,0))
plt.fill_between(Ns, np.mean(tau_A,0) - np.std(tau_A,0), np.mean(tau_A,0) + np.std(tau_A,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{c} (\tau_A)$')

# Show plot
plt.savefig("beta_diff_const_tA_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
