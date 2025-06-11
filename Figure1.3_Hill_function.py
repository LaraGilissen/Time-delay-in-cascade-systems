import numpy as np
import matplotlib.pyplot as plt

K = 1
x = np.linspace(0, 2.5, 1000)
ms = [1, 5, 10, 100, 1000]

def Hill(x, K, m):
    return K**m / (K**m + x**m)

plt.figure(figsize=(4, 2))

for i, m in enumerate(ms):
    y = Hill(x, K, m)
    plt.plot(x, y, label=fr'$m$={m}', color='tab:blue', alpha=(i+1)/5)

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.xticks([0, K], ['0', r'$K$'])
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0, 2.5)
plt.ylim(0, 1.1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Hill_function.pdf", bbox_inches = 'tight')
plt.show()
