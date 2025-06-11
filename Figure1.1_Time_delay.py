import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc

t = np.linspace(0, 20, 1000)

def x_in(t):
    input = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if 0 <= t_val <= 3:
            input[i] = 1
        else:
            input[i] = 0
    return input

def x_out_discr(t):
    return x_in(t-5)

def x_out_distr(t):
    output = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if t_val <= 3:
            output[i] = gammainc(5, t_val)
        else:
            output[i] = (gammainc(5, t_val) - gammainc(5, (t_val - 3)))
    return output

plt.figure(figsize=(3.2, 2))
plt.plot(t, x_in(t), label=r'$x_{in}$')
plt.plot(t, x_out_discr(t), label=r'$x_{out}$' + '\ndiscrete')
plt.plot(t, x_out_distr(t), label=r'$x_{out}$' + '\ndistributed')
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.xticks([])
plt.yticks([])
plt.xlim(0, 20)
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("Time_delay.pdf", bbox_inches = 'tight')
plt.show()