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

# Define parameters
t0 = 5

def G_N(t, betas):
    N = len(betas)
    total = 0.0
    for k in range(N):
        logprod = 0
        for l in range(N):
            if l != k:
                logprod += np.log(betas[l] - betas[k])
        total += np.exp(-betas[k] * t) / np.exp(logprod)
    return total

def eq(t, beta, t0):
    if t <= t0:
        equation = G_N(t, beta)
    else:
        equation = G_N(t, beta) - G_N(t-t0, beta)
    return equation

def find_tau_m(betas, t0, u0):
    result = fsolve(lambda t: eq(t, betas, t0), x0=u0)[0]
    return result

def find_tau_g(betas, t0):
    N = len(betas)
    result = t0 / 2
    for k in range(N):
        result += 1 / betas[k]
    return result

tau_m = np.zeros((len(cs), len(Ns)))
tau_g = np.zeros((len(cs), len(Ns)))

for i, c in enumerate(cs):
    for j, N in enumerate(Ns):
        tau_ms = []
        tau_gs = []
        for _ in range(runs):
            # Generate N beta values from U(c-w/2, c+w/2)
            betas = np.random.uniform(c - w/2, c + w/2, size=N)
            u0 = N/c + t0/2
            if N == 1: # The maximum is equal to t0 when N = 1 (x_1 is not differentiable in t0)
                tau_m_result = t0
            else:
                tau_m_result = find_tau_m(betas, t0, u0)
            tau_ms.append(tau_m_result)
            tau_g_result = find_tau_g(betas, t0)
            tau_gs.append(tau_g_result)
        tau_m[i, j] = trim_mean(tau_ms, 0.05)
        tau_g[i, j] = trim_mean(tau_gs, 0.05)


#########
# Plots #
#########

N, c = np.meshgrid(Ns, cs)

# Tau_m and tau_g #############################################################################################
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True, constrained_layout=True)
c_m = ax[0].pcolor(N, c, np.log10(tau_m), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)
c_g = ax[1].pcolor(N, c, np.log10(tau_g), cmap='viridis', shading='auto', vmin=-1.3, vmax=3)

# Labels
ax[0].tick_params(axis='both', which='major')
ax[0].set_xlabel(r'Number of states $N$')
ax[0].set_ylabel('Centre of interval \n coefficients $c$')
ax[1].tick_params(axis='both', which='major')
ax[1].set_xlabel(r'Number of states $N$')
ax[1].set_ylabel('Centre of interval \n coefficients $c$')

# Colorbar
cbar = fig.colorbar(c_m, ax=ax.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.03)
cbar.set_label(r'log$_{10}(\tau)$')
cbar.ax.tick_params()

# Save and show plot
plt.savefig("beta_diff_pulse.pdf", bbox_inches = 'tight')
plt.show()


# N = 10 ######################################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, tau_m[:,9], label=r'$\tau_m$')
plt.plot(cs, tau_g[:,9], label=r'$\tau_g$')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'Time delay $\tau$')
plt.legend()

# Show plot
plt.savefig("beta_diff_pulse_c.pdf", bbox_inches = 'tight')
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
plt.savefig("beta_diff_pulse_N.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, np.nanmean(tau_m,1))
plt.fill_between(cs, np.nanmean(tau_m,1) - np.nanstd(tau_m,1), np.nanmean(tau_m,1) + np.nanstd(tau_m,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'$mean_N (\tau_m)$')

# Show plot
plt.savefig("beta_diff_pulse_tm_c_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_m over c ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_m,0))
plt.fill_between(Ns, np.mean(tau_m,0) - np.std(tau_m,0), np.mean(tau_m,0) + np.std(tau_m,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{c} (\tau_m)$')

# Show plot
plt.savefig("beta_diff_pulse_tm_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over N ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(cs, np.mean(tau_g,1))
plt.fill_between(cs, np.mean(tau_g,1) - np.std(tau_g,1), np.mean(tau_g,1) + np.std(tau_g,1), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Centre of interval coefficients $c$')
plt.ylabel(r'$mean_N (\tau_g)$')

# Show plot
plt.savefig("beta_diff_pulse_tg_c_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# Mean tau_g over c ###########################################################################################
plt.figure(figsize=(3.1, 2))
plt.plot(Ns, np.mean(tau_g,0))
plt.fill_between(Ns, np.mean(tau_g,0) - np.std(tau_g,0), np.mean(tau_g,0) + np.std(tau_g,0), color='#888888')
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Number of states $N$')
plt.ylabel(r'$mean_{c} (\tau_g)$')

# Show plot
plt.savefig("beta_diff_pulse_tg_N_mean.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()
