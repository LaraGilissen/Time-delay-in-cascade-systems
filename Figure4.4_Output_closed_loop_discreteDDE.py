import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
 
# Parameters
alpha = 1
beta = 1
K = 0.5
t = np.linspace(0, 50, 1000)
 
# Define the discrete DDE system
def make_model(K, m, alpha, beta, tau):
    def model(Y, t):
        x = Y(t)
        x_tau = Y(t - tau) if t - tau >= 0 else hist(t - tau)
        f_x_tau = K**m / (K**m + x_tau**m)
        return alpha * f_x_tau - beta * x
    return model
 
# History function (for t ≤ 0)
def hist(t):
    return 0

def calc_output(t, K, m, alpha, beta, N):
    tau = (N-1) / beta
    model = make_model(K, m, alpha, beta, tau)
    output = ddeint(model, hist, t)
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
plt.savefig("Closed_loop_discrete_output_m2_N2.pdf", bbox_inches = 'tight')
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
plt.savefig("Closed_loop_discrete_output_m5_N3.pdf", bbox_inches = 'tight')
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
plt.savefig("Closed_loop_discrete_output_m10_N5.pdf", bbox_inches = 'tight')
plt.tight_layout()
plt.show()