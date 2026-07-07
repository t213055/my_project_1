import numpy as np
import itertools
from scipy import integrate

N_v = 2
N_h = 2
states = [-1.0, 1.0]

H_matrix = np.array(list(itertools.product(states, repeat=N_h)))

beta = 1.0
n = N_v
m = N_h

a = (beta**2) / (2*(n + m))

#レプリカ1の可視, 隠れ変数
v1 = np.array([-0.2, 0.3])
h1 = np.array([-1.0, 1.0])

#レプリカ2の可視, 隠れ変数
v2 = np.array([0.4, -0.2])
h2 = np.array([1.0, -1.0])

Befor = np.exp(
    a * ((v1[0]*h1[0] + v2[0]*h2[0])**2 + (v1[1]*h1[0] + v2[1]*h2[0])**2 + (v1[0]*h1[1] + v2[0]*h2[1])**2 + (v1[1]*h1[1] + v2[1]*h2[1])**2)
    )
print("a :", ((v1[0]*h1[0] + v2[0]*h2[0])**2 + (v1[1]*h1[0] + v2[1]*h2[0])**2 + (v1[0]*h1[1] + v2[0]*h2[1])**2 + (v1[1]*h1[1] + v2[1]*h2[1])**2))
print("Before", Befor)

After = np.exp(
    a * (m *(v1[0]**2 + v1[1]**2 + v2[0]**2 + v2[1]**2)
        + 2 * (v1[0]*v2[0] + v1[1]*v2[1]) * (h1[0]*h2[0] + h1[1]*h2[1]))
    )
print("a :", (m *(v1[0]**2 + v1[1]**2 + v2[0]**2 + v2[1]**2) + 2 * (v1[0]*v2[0] + v1[1]*v2[1]) * (h1[0]*h2[0] + h1[1]*h2[1])))
print("After", After)


def integrand_v(v11, v12):
    return 

Before = 