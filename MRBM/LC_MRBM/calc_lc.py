import numpy as np
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

def gaussian_pdf(z):
    return (np.sqrt(2 * np.pi))**-1 * np.exp(-0.5 * z**2)

def visible_variable_array(t):
    k_start = (t // 2) + 1
    k = np.arange(t, k_start - 1, -1)
    v_k = (2 * k - t) / t
    return v_k

def SP(beta, q, q_hat, T_alpha, alpha, v_k, b, c, delta_term, tol_sp):
    def integrand_q_v(z):
        B_z = b + z * np.sqrt(q_hat[0])
        N_z = np.sum(v_k * np.exp(A * v_k**2) * np.sinh(B_z * v_k))
        D_z = 0.5 * delta_term + np.sum(np.exp(A * v_k**2) * np.cosh(B_z * v_k))
        f_z = N_z / D_z
        return gaussian_pdf(z) * f_z**2

    def integrand_q_h(z):
        f_z = np.tanh(c + z * np.sqrt(q_hat[1]))
        return gaussian_pdf(z) * f_z**2

    iteration = 0
    while True:
        q_old = q.copy()
        q_hat_old = q_hat.copy()
        
        A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_hat[0])
        
        q[0], _ = quad(integrand_q_v, -12, 12)
        q[1], _ = quad(integrand_q_h, -12, 12)
        q_hat = beta**2 * T_alpha @ q
        iteration += 1
        
        if (np.all(np.abs(q - q_old) <= tol_sp) and np.all(np.abs(q_hat - q_hat_old) <= tol_sp)):
            return q, q_hat, A, iteration

def Effective_Susceptibility_Matrices(q_hat, A, v_k, b, c, delta_term):
    v_k_row = v_k[np.newaxis, :]
    
    deg = 50
    x_i, w_i = hermgauss(deg)
    z_i = np.sqrt(2) * x_i
    weights = w_i / np.sqrt(np.pi)

    B_i = b + z_i * np.sqrt(q_hat[0])
    B_i = B_i[:, np.newaxis] 
    
    D = delta_term + 2 * np.sum(np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    N_1 =  2 * np.sum(v_k_row * np.exp(A * v_k_row**2) * np.sinh(B_i * v_k_row), axis=1)
    N_2 =  2 * np.sum(v_k_row**2 * np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    
    Ev1 = N_1 / D
    Ev2 = N_2 / D
    Eh1 = np.tanh(c + z_i * np.sqrt(q_hat[1]))
    
    V00 = np.sum((Ev2 - Ev1**2) * weights)
    V11 = np.sum((1 - Eh1**2) * weights)
    V = np.array([[V00, 0], [0, V11]])

    U00 = np.sum((Ev2 * Ev1 - Ev1**3) * weights)
    U11 = np.sum((Eh1 - Eh1**3) * weights)
    U = np.array([[U00, 0], [0, U11]])

    W00 = np.sum((Ev2**2 - 4 * Ev2 * Ev1**2 + 3 * Ev1**4) * weights)
    W11 = np.sum((1 - 4 * Eh1**2 + 3 * Eh1 **4) * weights)
    W = np.array([[W00, 0], [0, W11]])
    
    return V, U, W

def Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha):
    M = np.identity(2) - beta**2 * T_alpha @ W
    B = 2 * beta**2 * T_alpha @ U
    HQ = np.linalg.solve(M, B)
    chi = (1 + alpha)**-1 * np.array([[1, 0], [0, alpha]]) @ (V - U @ HQ)
    return chi
