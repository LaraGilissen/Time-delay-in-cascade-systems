import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import gamma as gamma_func
 
# Parameters
alpha = 1
beta = 1
K = 0.5
t = np.linspace(0, 50, 1000)
 
# Define G_N
def G_N(s, alpha, beta, N):
    return alpha**N / (gamma_func(N)) * s**(N - 1) * np.exp(-beta * s) 
 
# Define the model
def make_model(K, m, alpha, beta, tau_vals, kernel):
    def model(x, t, past_t, past_x):
        if t < tau_vals[-1]:
            x_past = np.array([np.interp(t - tau, past_t, past_x) if (t - tau) >= 0 else hist(t - tau) for tau in tau_vals])
        else:
            x_past = np.interp(t - tau_vals, past_t, past_x)
        x_eff = np.trapz(kernel * x_past, tau_vals)
        f_x_eff = K**m / (K**m + x_eff**m)
        return alpha * f_x_eff - beta * x
    return model
 
# History function (for t ≤ 0)
def hist(t):
    return 0

def calc_output(t, K, m, alpha, beta, N):
    tau_mean = (N-1) / beta
    tau_vals = np.linspace(0, 10 * tau_mean, 500)
    kernel = G_N(tau_vals, alpha, beta, N-1)
    output = np.zeros_like(t)
    output[0] = hist(0)
    for i in range(1, len(t)):
        tspan = [t[i - 1], t[i]]
        past_t = t[:i]
        past_x = output[:i]
        model = make_model(K, m, alpha, beta, tau_vals, kernel)
        output[i] = odeint(model, output[i - 1], tspan, args=(past_t, past_x))[-1, 0]
    return output


#########
# Plots #
#########

# m = 2, N = 2 ################################################################################################
m = 2
N = 2
plt.figure(figsize=(3.1, 2))
plt.plot(t, calc_output(t, K, m, alpha, beta, N))
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x(t)$')

# Show plot
plt.savefig("Closed_loop_distributed_output_m2_N2.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# m = 5, N = 3 ################################################################################################
m = 5
N = 3
plt.figure(figsize=(3.1, 2))
plt.plot(t, calc_output(t, K, m, alpha, beta, N))
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x(t)$')

# Show plot
plt.savefig("Closed_loop_distributed_output_m5_N3.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()


# m = 10, N = 5 ##############################################################################################
m = 10
N = 5
plt.figure(figsize=(3.1, 2))
plt.plot(t, calc_output(t, K, m, alpha, beta, N))
plt.tick_params(axis='both', which='major')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Output $x(t)$')

# Show plot
plt.savefig("Closed_loop_distributed_output_m10_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()