import numpy as np
import itertools
from scipy import integrate

#構造と隠れ変数の実現値
N_v = 2
N_h = 2
states = [-1.0, 1.0]

#パラメータ
b = 1.5
c = -2.0
gamma = np.log(np.exp(1) - 1)

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

def softplus(x):
    return np.log(1 + np.exp(x))
"""
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
"""
#可視変数（連続値）と隠れ変数（Ising）
v = np.array([[-0.2, 0.3],[0.4, -0.2]])
h = np.array([[-1.0, 1.0],[1.0, -1.0]])
#サイトごとに独立である場合の、レプリカインデックスのみに依存する、可視変数と隠れ変数
v = np.array([[-0.8, -0.8],[0.4, 0.4]])
h = np.array([[1.0, 1.0],[-1.0, -1.0]])

#秩序パラメータ　適当な値を設定
qv = 1.5
qh = -3.0
rv = np.array([2.5, 2.5])
hqv = 2.0
hqh = 0.5
hrv = np.array([1.2, 1.2])

"""
print("p.2 最後 :")
First = n*m*a*np.sum(rv) + 2*n*m*a*(qv*qh) - n*(rv.T @ hrv) - n*(qv*hqv) - m*(qh*hqh)
Second = - 1/(2*softplus(gamma))*(np.sum(v**2)) + b*(np.sum(v)) + hrv[0]*(np.sum(v[0, :]**2)) + hrv[1]*(np.sum(v[1, :]**2)) + hqv*(np.prod(v[:, 0]) + np.prod(v[:, 1]))
Third = c*np.sum(h) + hqh*(np.prod(h[:, 0]) + np.prod(h[:, 1]))
print('{:.6g}'.format(First), '{:.6g}'.format(Second), '{:.6g}'.format(Third))
"""

print("p.3 式A.3 :")
alpha = m / n
First = n *((alpha*beta**2)/(2*(1+alpha)) * np.sum(rv) + (alpha*beta**2)/(1+alpha)*qv*qh - rv.T @ hrv - qv*hqv - alpha*qh*hqh)
Second = n * ((-(2*softplus(gamma))**(-1)*np.sum(v[:, 0]**2)) + b*np.sum(v[:, 0]) + (hrv[0]*np.average(v[0])**2 + hrv[1]*np.average(v[1])**2) + hqv*np.average(v[0])*np.average(v[1]))
Third = n * alpha *(c * (np.average(h[0]) + np.average(h[1])) + hqh*np.average(h[0])*np.average(h[1]))
print('{:.6g}'.format(First), '{:.6g}'.format(Second), '{:.6g}'.format(Third))

#自己相関秩序→各レプリカで全て同じ値（＊パラメータが同じ→同じ熱揺らぎ→無限系で大数の法則により熱揺らぎが消失→全て同じ）
#スピングラス秩序→各レプリカ間で全て同じ値(＊レプリカ対称性を仮定→レプリカ間で違いはない→全て同じ)
print("p.3 式A.4 : ")
x = 2
First = x * n *(alpha*beta**2/(2*(1+alpha)) * np.average(rv) + ((x-1)*alpha*beta**2)/(2*(1+alpha))*qv*qh - np.average(rv)*np.average(hrv) - (x-1)/2*qv*hqv + alpha*(hqh/2*((1-x)*qh)-1))
Second = n * ((-(2*softplus(gamma))**(-1)*np.sum(v[:, 0]**2)) + b*np.sum(v[:, 0]) + (hrv[0]*np.average(v[0])**2 + hrv[1]*np.average(v[1])**2) + hqv/2*(np.sum(v[:,0])**2 - np.sum(v[:,1]**2)))
Third = n * alpha*(c*(np.average(h[0]) + np.average(h[1])) + 0.5*(np.sqrt(hqh)*(np.average(h[0]) + np.average(h[1])))**2)
print('{:.6g}'.format(First), '{:.6g}'.format(Second), '{:.6g}'.format(Third))

#A.3 → A.4への変形が合っていると考えて次のステップへ進む
