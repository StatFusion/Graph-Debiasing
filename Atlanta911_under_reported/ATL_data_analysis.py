import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cvxpy as cp


data = loadmat('Atlanta911_under_reported/atl_data.mat')
A = data['A']
y = data['y'].flatten() / 500
X = data['X']

# Parameters
m = 78
K = 3
p_min = 0.1
p_max = 0.9

# Graph Laplacian
A = (A + A.T) / 2
A = np.where(A != 0, 1, 0)

A = A - np.diag(np.diag(A))
D = np.sum(A, axis=1)
L = np.diag(D) - A

lambda_1 = 0.001
lambda_2 = 0.001

X = np.hstack((np.ones((m, 1)), X))
K += 1
H = np.eye(m) - X @ np.linalg.pinv(X.T @ X) @ X.T

# Check if matrices are symmetric
H_is_symmetric = np.allclose(H, H.T)
L_is_symmetric = np.allclose(L, L.T)

# Check if matrices are PSD by computing eigenvalues
H_eigenvals = np.linalg.eigvals(H)
L_eigenvals = np.linalg.eigvals(L)

H_is_psd = np.all(H_eigenvals >= -1e-10)  # Allow for small numerical errors
L_is_psd = np.all(L_eigenvals >= -1e-10)

print("\nMatrix Properties:")
print(f"H is symmetric: {H_is_symmetric}")
print(f"L is symmetric: {L_is_symmetric}")
print(f"H is PSD: {H_is_psd}")
print(f"L is PSD: {L_is_psd}")

print("\nSmallest eigenvalues:")
print(f"H min eigenvalue: {np.min(H_eigenvals):.2e}")
print(f"L min eigenvalue: {np.min(L_eigenvals):.2e}")


# Initialization
iter = 100000
loss = np.zeros(iter)

p_level = 0.8

# v_est = 1-p_est
v_est = np.ones(m)
n_est = y / v_est

step = 0.001
inner_iter = 10

# Iterations
for ii in range(iter):
    for _ in range(inner_iter):
        dn1 = -2 * (y - n_est * v_est) * v_est / m
        dn2 = 2 * lambda_2 * H @ n_est
        n_est = n_est - step * (dn1 + dn2)
        n_est[n_est < 1] = 1

    for _ in range(inner_iter):
        dp1 = -2 * (y - n_est * v_est) * n_est / m
        dp2 = 2 * lambda_1 * L @ v_est
        v_est = v_est - step * (dp1 + dp2)
        v_est[v_est > p_max] = p_max
        v_est[v_est < p_min] = p_min
    loss[ii] = np.linalg.norm(y - n_est * v_est) ** 2 / m + lambda_1 * v_est.T @ L @ v_est + lambda_2 * n_est.T @ H @ n_est
p_est = 1-v_est

# Visualization
plt.figure()
plt.plot(loss, '-')
plt.title('Loss')
plt.grid(True)
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(p_est, label='est p')
plt.legend()
plt.subplot(2, 1, 2)
plt.stem(n_est, label='est n')
plt.stem(y, 'g', label='y')
plt.legend()
plt.show()

print('n, y, n_est:')
print(np.column_stack((y, n_est)))