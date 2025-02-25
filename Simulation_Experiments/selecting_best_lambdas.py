import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Cross-Validation Code for Selecting λ₁ and λ₂
# ============================================================

# Thresholds for v^T L v /m < threshold
# True/False refers to low/high variance

thresholds_dict_complex = {
    25:  {True: 0.012, False: 0.050},
    50:  {True: 0.050, False: 0.175},
    75:  {True: 0.090, False: 0.320},
    100: {True: 0.135, False: 0.525}
}

thresholds_dict_simple = {
    25:  {True: 0.005, False: 0.020},
    50:  {True: 0.015, False: 0.060},
    75:  {True: 0.028, False: 0.100},
    100: {True: 0.043, False: 0.168}
}

def run_simulation(lambda_1, lambda_2, threshold, complex_graph, m, low_var, large_n, outer_iter=1000):
    while True:
        if complex_graph:
            A = (np.random.rand(m, m) > 0.7).astype(float)
            A = ((A + A.T) / 2 > 0).astype(float)
            np.fill_diagonal(A, 0)
            if np.all(np.sum(A, axis=1) > 0):
                break
        else:
            A = (np.random.rand(m, m) > 0.9).astype(float)
            A = ((A + A.T) / 2 > 0).astype(float)
            np.fill_diagonal(A, 0)
            if np.all(np.sum(A, axis=1) > 0): 
                break

    if complex_graph:
        weights = [0.2, 0.4, 0.6, 0.8, 1.0]
        K = 8
    else:
        weights = [0.4, 0.6]
        K = 3

    W = np.zeros_like(A)
    mask_upper = np.triu(A, k=1)
    random_weights = np.random.choice(weights, size=int(np.sum(mask_upper)))
    W[mask_upper > 0] = random_weights
    W = W + W.T
    D = np.sum(W, axis=1)
    L = np.diag(D) - W

    if low_var:
        p_var = 0.1
        p_level = 0.3
    else:
        p_var = 0.2
        p_level = 0.4

    while True:
        p = p_level + np.random.randn(m) * p_var
        p = np.clip(p, 0.05, 0.95)
        p = 0.05 + p * (0.95 - 0.05)
        if (p.T @ L @ p) / m < threshold:
            break

    # --- Simulate X and n ---
    while True:
        if complex_graph:
            if large_n:
                X = 1.5 * np.random.randn(m, K) + (1.0 if m < 50 else 1.5)
                my_beta = np.random.choice([0.25, 0.5, 0.75, 1.0], size=K)
            else:
                X = np.random.randn(m, K) + (1.0 if m < 50 else 1.5)
                my_beta = np.random.choice([0.125, 0.25, 0.375, 0.5], size=K)
        else:
            X = np.random.randn(m, K) + (2.0 if m < 50 else 2.5)
            my_beta = np.ones(K)*0.5

        n = np.round(np.exp(X @ my_beta))
        if np.all(n >= 10):
            break


    trial = 10
    y = np.mean([n + np.random.poisson(n * p) for _ in range(trial)], axis=0)    
    H = np.eye(m) - X @ np.linalg.pinv(X.T @ X) @ X.T
    n_est = y.copy()
    p_est = np.ones(m) * 1.0  # p_est represents 1+p

    step = 0.4
    inner_iter = 1

    for ii in range(outer_iter):
        for _ in range(inner_iter):
            du = 2 * (-np.log(y) + (np.log(n_est) + np.log(p_est)) + lambda_2 * H @ np.log(n_est))
            n_est = np.exp(np.log(n_est) - step * du)

        for _ in range(inner_iter):
            dv = 2 * (-np.log(y) + (np.log(n_est) + np.log(p_est)) + lambda_1 * L @ np.log(p_est))
            p_est = np.exp(np.log(p_est) - step * dv)
    
    final_l1_err_p = np.mean(np.abs(p_est-1 - p))
    return final_l1_err_p

# lambda1_candidates = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025,
#                       0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04]
lambda1_candidates = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]
lambda2_candidates = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6]


n_cv_rounds = 10
cv_errors = np.zeros((len(lambda1_candidates), len(lambda2_candidates)))


m_values = [25, 50, 75, 100]
low_var_options = [True, False]
large_n_options = [True, False]
columns = pd.MultiIndex.from_product([low_var_options, large_n_options],
                                      names=['low_var', 'large_n'])
best_lambdas_complex = pd.DataFrame(index=m_values, columns=columns)
best_lambdas_simple  = pd.DataFrame(index=m_values, columns=columns)


print("Complex Graph")
for m in [50, 75, 100]:
    print(f"m = {m}")
    for low_var in [True, False]:
        print(f"low_var = {low_var}")
        for large_n in [True, False]:
            print(f"large_n = {large_n}")
            for i, lam1 in enumerate(lambda1_candidates):
                for j, lam2 in enumerate(lambda2_candidates):
                    print(f"λ₁: {lam1}, λ₂: {lam2}")
                    errors = []
                    for cv_round in range(n_cv_rounds):
                        np.random.seed(cv_round) # for reproducibility
                        err = run_simulation(lambda_1=lam1, lambda_2=lam2, threshold=thresholds_dict_complex[m][low_var], complex_graph=True, m=m, low_var=low_var, large_n=large_n, outer_iter=1500)
                        # print("one simulation done")
                        errors.append(err)
                    avg_error = np.mean(errors)
                    cv_errors[i, j] = avg_error
                    print(f"λ₁: {lam1}, λ₂: {lam2} -> CV Error: {avg_error:.4f}")
            cv_errors[np.isnan(cv_errors)] = 1
            best_idx = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
            best_lambda1 = lambda1_candidates[best_idx[0]]
            best_lambda2 = lambda2_candidates[best_idx[1]]
            best_lambdas_complex.loc[m, (low_var, large_n)] = (best_lambda1, best_lambda2, cv_errors[best_idx])
            print(f"Best λ_1: {best_lambda1}, Best λ_2: {best_lambda2}, Best CV Error: {cv_errors[best_idx]:.4f}")
            best_lambdas_complex.to_csv('Simulation_Experiments/best_lambdas_complex.csv', index=True)
            best_lambdas_complex.to_pickle("Simulation_Experiments/best_lambdas_complex.pkl")


print("Simple Graph")
for m in [25, 50, 75, 100]:
    print(f"m = {m}")
    for low_var in [True, False]:
        print(f"low_var = {low_var}")
        large_n = False
        print(f"large_n = {large_n}")
        for i, lam1 in enumerate(lambda1_candidates):
            for j, lam2 in enumerate(lambda2_candidates):
                errors = []
                for cv_round in range(n_cv_rounds):
                    np.random.seed(cv_round) # for reproducibility
                    err = run_simulation(lambda_1=lam1, lambda_2=lam2, threshold=thresholds_dict_simple[m][low_var], complex_graph=False, m=m, low_var=low_var, large_n=large_n, outer_iter=1500)
                    errors.append(err)
                avg_error = np.mean(errors)
                cv_errors[i, j] = avg_error
                print(f"λ₁: {lam1}, λ₂: {lam2} -> CV Error: {avg_error:.4f}")
        cv_errors[np.isnan(cv_errors)] = 1
        best_idx = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
        best_lambda1 = lambda1_candidates[best_idx[0]]
        best_lambda2 = lambda2_candidates[best_idx[1]]
        best_lambdas_simple.loc[m, (low_var, large_n)] = (best_lambda1, best_lambda2, cv_errors[best_idx])
        print(f"Best λ_1: {best_lambda1}, Best λ_2: {best_lambda2}, Best CV Error: {cv_errors[best_idx]:.4f}")
        best_lambdas_simple.to_csv('Simulation_Experiments/best_lambdas_simple.csv', index=True)
        best_lambdas_simple.to_pickle("Simulation_Experiments/best_lambdas_simple.pkl")
print(best_lambdas_complex)
print(best_lambdas_simple)