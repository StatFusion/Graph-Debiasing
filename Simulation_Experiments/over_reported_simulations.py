import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thresholds for v^T L v /m < threshold
# True/False refers to low/high variance

thresholds_dict_complex = {
    25:  {True: 0.012, False: 0.050},
    50:  {True: 0.050, False: 0.175},
    75:  {True: 0.090, False: 0.320},
    100: {True: 0.135, False: 0.525},
    125: {True: 0.185, False: 0.700},
    150: {True: 0.235, False: 0.880},
}

thresholds_dict_simple = {
    25:  {True: 0.005, False: 0.020},
    50:  {True: 0.015, False: 0.060},
    75:  {True: 0.028, False: 0.100},
    100: {True: 0.043, False: 0.168},
    125: {True: 0.068, False: 0.200},
    150: {True: 0.080, False: 0.220},
}


# These best lambdas are pre-selected by cross-validation.
best_lambdas_complex = pd.read_pickle('Simulation_Experiments/best_lambdas_complex.pkl')
best_lambdas_simple = pd.read_pickle('Simulation_Experiments/best_lambdas_simple.pkl')

rounds = 500
if rounds == 500:
    np.random.seed(42)
m = 150
outer_iter = 5000
complex_graph = True
low_var = True
large_n = True
threshold = thresholds_dict_complex[m][low_var]

lambda_1, lambda_2, _ = best_lambdas_complex.loc[m][(low_var, large_n)]


loss = np.zeros(outer_iter)
relative_err_n = np.zeros(outer_iter)
relative_err_p = np.zeros(outer_iter)
l1_err_p = np.zeros(outer_iter)
linf_err_p = np.zeros(outer_iter)
linf_err_n = np.zeros(outer_iter)

loss_rounds = np.zeros((rounds,outer_iter))
relative_err_n_rounds = np.zeros((rounds,outer_iter))
linf_err_n_rounds = np.zeros((rounds,outer_iter))
relative_err_p_rounds = np.zeros((rounds,outer_iter))
l1_err_p_rounds = np.zeros((rounds,outer_iter))
linf_err_p_rounds = np.zeros((rounds,outer_iter))



for tt in range(rounds):
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

    # Generate X and n
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
    #y = np.mean([n + np.random.binomial(n.astype(int), p) for _ in range(trial)], axis=0)
    H = np.eye(m) - X @ np.linalg.pinv(X.T @ X) @ X.T

    # Initializations
    n_est = y
    v_est = np.zeros(m)+1.0  # v_est represents 1+p
    step = 0.4
    inner_iter = 1

    for ii in range(outer_iter):
        relative_err_n[ii] = np.sum(np.abs(n_est - n)) / np.sum(n)
        relative_err_p[ii] = np.sum(np.abs(v_est-1 - p)) / np.sum(p)
        l1_err_p[ii] = np.mean(np.abs(v_est-1 - p))

        linf_err_n[ii] = np.max(np.abs(n_est - n))
        linf_err_p[ii] = np.max(np.abs(v_est-1 - p))

        # training loss
        loss[ii] = np.linalg.norm(y - n_est * (v_est-1)) ** 2 + lambda_1 * (np.log(v_est)).T @ L @ (np.log(v_est)) + lambda_2 * np.log(n_est).T @ H @ np.log(n_est)
        if np.isnan(loss[ii]) and ii > 0:
            loss[ii] = loss[ii-1]
        if ii == 1:
            loss[0] = loss[ii]
            loss_rounds[tt,0] = loss[0]

        loss_rounds[tt,ii] = loss[ii]
        relative_err_n_rounds[tt,ii] = relative_err_n[ii]
        relative_err_p_rounds[tt,ii] = relative_err_p[ii]
        linf_err_n_rounds[tt,ii] = linf_err_n[ii]
        linf_err_p_rounds[tt,ii] = linf_err_p[ii]
        l1_err_p_rounds[tt,ii] = l1_err_p[ii]
    
        for _ in range(inner_iter):
            du = 2 * (-np.log(y) + (np.log(n_est) + np.log(v_est)) + lambda_2 * H @ np.log(n_est))
            n_est = np.exp(np.log(n_est) - step * du)
        for _ in range(inner_iter):
            dv = 2 * (-np.log(y) + (np.log(n_est) + np.log(v_est)) + lambda_1 * L @ np.log(v_est))
            v_est = np.exp(np.log(v_est) - step * dv)

    if tt % 10 == 0:
        print(f"Round {tt+1} completed")

if low_var:
    if large_n:
        results_dir = "Simulation_Experiments/new_over_reported/Low_Var_p/Large_n"
    else:
        results_dir = "Simulation_Experiments/new_over_reported/Low_Var_p/Small_n"
else:
    if large_n:
        results_dir = "Simulation_Experiments/new_over_reported/High_Var_p/Large_n"
    else:
        results_dir = "Simulation_Experiments/new_over_reported/High_Var_p/Small_n"


plt.rcParams.update({
    "font.size": 12,           # Larger font size
    "figure.figsize": (6, 4),  # Consistent figure size
    "axes.grid": True,         # Add grid to plots
    "grid.alpha": 0.5,         # Transparency for grid
    "savefig.dpi": 300,        # High-resolution output
    "lines.linewidth": 2,      # Thicker lines
})

if rounds > 1:
    print("Over-reported")
    print("m:", m)
    print("Complex Graph:", complex_graph)
    print("Large n:", large_n)
    print("Low variance:", low_var)

    print(f"Average loss: {np.nanmean(loss_rounds)}+-{np.nanstd(loss_rounds)}")
    print(f"Average relative L1 error of n: {np.nanmean(relative_err_n_rounds[:,-1])}+-{np.nanstd(relative_err_n_rounds[:,-1])}")
    print(f"Average relative L1 error of p: {np.nanmean(relative_err_p_rounds[:,-1])}+-{np.nanstd(relative_err_p_rounds[:,-1])}")
    print(f"Average L1 error of p: {np.nanmean(l1_err_p_rounds[:,-1])}+-{np.nanstd(l1_err_p_rounds[:,-1])}")
    print(f"Average Linf error of p: {np.nanmean(linf_err_p_rounds[:,-1])}+-{np.nanstd(linf_err_p_rounds[:,-1])}")
    np.save(f"{results_dir}/rounds_loss_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", loss_rounds)
    np.save(f"{results_dir}/rounds_relative_err_n_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", relative_err_n_rounds)
    np.save(f"{results_dir}/rounds_relative_err_p_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", relative_err_p_rounds)
    np.save(f"{results_dir}/rounds_l1_err_p_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", l1_err_p_rounds)
    np.save(f"{results_dir}/rounds_linf_err_p_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", linf_err_p_rounds)
    np.save(f"{results_dir}/rounds_linf_err_n_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", linf_err_n_rounds)
    
    # Relative L1 error of n
    plt.figure()
    mean_err_n = np.nanmean(relative_err_n_rounds, axis=0)
    plt.plot(mean_err_n, label=r"Mean Relative Error of $\mathrm{n}_{\mathrm{est}}$", color="blue", linestyle="--")
    plt.ylabel(r"Relative $\mathrm{L}_\mathrm{1}$ Error of $\mathrm{n}_{\mathrm{est}}$")
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.title(r"Convergence of Relative $\mathrm{n}_{\mathrm{est}}$ Error Across 100 Rounds")
    final_n = mean_err_n[-1]
    plt.annotate(f'{final_n:.2f}', xy=(len(mean_err_n) - 1, final_n), 
                xytext=(len(mean_err_n) - 1.5, final_n + 0.05),
                arrowprops=dict(arrowstyle='-', color='blue'),
                fontsize=10, color="blue")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/rounds_n_error_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/rounds_n_error_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    # Relative L1 error of p
    plt.figure()
    mean_err_p = np.nanmean(relative_err_p_rounds, axis=0)
    plt.plot(mean_err_p, label=r"Mean Relative Error of $\mathrm{p}_{\mathrm{est}}$", color="red", linestyle="--")
    plt.ylabel(r"Relative $\mathrm{L}_\mathrm{1}$ Error of $\mathrm{p}_{\mathrm{est}}$")
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.title(r"Convergence of Relative $\mathrm{p}_{\mathrm{est}}$ Error Across 100 Runs")
    final_p = mean_err_p[-1]
    plt.annotate(f'{final_p:.2f}', xy=(len(mean_err_p) - 1, final_p), 
                xytext=(len(mean_err_p) - 1.5, final_p + 0.05),
                arrowprops=dict(arrowstyle='-', color='red'),
                fontsize=10, color="red")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/rounds_p_error_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/rounds_p_error_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    # L1 error of p
    plt.figure()
    mean_err_p = np.nanmean(l1_err_p_rounds, axis=0)
    plt.plot(mean_err_p, label=r"Mean $\mathrm{L}_\mathrm{1}$ Error of $\mathrm{p}_{\mathrm{est}}$", color="red", linestyle="--")
    plt.ylabel(r"$\mathrm{L}_\mathrm{1}$ Error of $\mathrm{p}_{\mathrm{est}}$")
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.title(r"Convergence of $\mathrm{L}_\mathrm{1}$ Error of $\mathrm{p}_{\mathrm{est}}$ Across 100 Runs")
    final_p = mean_err_p[-1]
    plt.annotate(f'{final_p:.2f}', xy=(len(mean_err_p) - 1, final_p), 
                xytext=(len(mean_err_p) - 1.5, final_p + 0.05),
                arrowprops=dict(arrowstyle='-', color='red'),
                fontsize=10, color="red")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/rounds_p_error_L1_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/rounds_p_error_L1_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    # Linf error of p
    plt.figure()
    mean_err_p = np.nanmean(linf_err_p_rounds, axis=0)
    plt.plot(mean_err_p, label=r"Mean $\mathrm{L}_\mathrm{inf}$ Error of $\mathrm{p}_{\mathrm{est}}$", color="red", linestyle="--")
    plt.ylabel(r"$\mathrm{L}_\mathrm{inf}$ Error of $\mathrm{p}_{\mathrm{est}}$")
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.title(r"Convergence of $\mathrm{L}_\mathrm{inf}$ Error of $\mathrm{p}_{\mathrm{est}}$ Across 100 Runs")
    final_p = mean_err_p[-1]
    plt.annotate(f'{final_p:.2f}', xy=(len(mean_err_p) - 1, final_p), 
                xytext=(len(mean_err_p) - 1.5, final_p + 0.05),
                arrowprops=dict(arrowstyle='-', color='red'),
                fontsize=10, color="red")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/rounds_p_error_Linf_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/rounds_p_error_Linf_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    # Loss
    scale_factor = int(np.floor(np.log10(max(loss))))  # Get the order of magnitude
    normalized_loss = np.array(loss) / (10 ** scale_factor)  # Normalize loss
    plt.figure()
    plt.plot(normalized_loss, label=r"Loss", color="green", linestyle="--")
    plt.ylabel(f"Loss ($\\times 10^{{{scale_factor}}}$)")  # Adjust label to indicate scale
    plt.xlabel("Iteration")
    plt.title("Convergence of Loss Across 100 Runs")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/rounds_loss_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/rounds_loss_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

else:
    if m > 25:
        plt.rcParams.update({
        "font.size": 12,           # Larger font size
        "figure.figsize": (8, 4),  # Consistent figure size
        "axes.grid": True,         # Add grid to plots
        "grid.alpha": 0.5,         # Transparency for grid
        "savefig.dpi": 300,        # High-resolution output
        "lines.linewidth": 2,      # Thicker lines
    })
    # Save results to files
    np.save(f"{results_dir}/loss_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", loss)
    np.save(f"{results_dir}/y_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", y)
    np.save(f"{results_dir}/n_est_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", n_est)
    np.save(f"{results_dir}/v_est_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", v_est)
    np.save(f"{results_dir}/relative_err_n_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", relative_err_n)
    np.save(f"{results_dir}/relative_err_p_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", relative_err_p)
    np.save(f"{results_dir}/l1_err_p_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.npy", l1_err_p)

    print(f"linf_err_p: {linf_err_p[-1]}")
    print(f"l1_err_p: {l1_err_p[-1]}")
    print(f"relative_err_n: {relative_err_n[-1]}")

    colors = ["blue", "red", "grey"]

    # "#D9C6A5"
    max_val = np.max([np.abs(n).max(), np.abs(n_est).max(), np.abs(y).max()])
    exponent = int(np.floor(np.log10(max_val)))  # e.g., if max_val is ~4500 then exponent = 3
    scale = 10 ** exponent
    plt.figure()
    x1 = np.arange(1, m + 1)
    x2 = np.arange(1, m + 1) + 0.25  # Shift right for true values
    x3 = np.arange(1, m + 1) - 0.25  # Shift left for observed y
    plt.vlines(x2, 0, n / scale, colors=colors[0], label=r'True $n$', linewidth=0.8)
    plt.vlines(x1, 0, n_est / scale, colors=colors[1], label=r'Estimated $n_{\mathrm{est}}$', linewidth=0.8)
    plt.vlines(x3, 0, y / scale, colors=colors[2], label=r'Observed $y$', linewidth=0.8)
    plt.scatter(x1, n_est / scale, color=colors[1], zorder=3, s=20)
    plt.scatter(x2, n / scale, color=colors[0], zorder=3, s=20)
    plt.scatter(x3, y / scale, color=colors[2], zorder=3, s=20)
    plt.xlabel("Node index")
    plt.ylabel(r"Value ($\times 10^{%d}$)" % exponent)
    plt.title(r"Comparison of $n$ and $n_{\mathrm{est}}$")
    plt.legend(loc="upper right")
    plt.savefig(results_dir+f"/n_comparison_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/n_comparison_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    plt.figure()
    x1 = np.arange(1, m+1) - 0.10  # Shift left for estimated values
    x2 = np.arange(1, m+1) + 0.10  # Shift right for true values
    plt.vlines(x2, 0, p, colors=colors[0], label=r'True $p$',linewidth=0.8)
    plt.vlines(x1, 0, v_est-1, colors=colors[1], label=r'Estimated $p_{\mathrm{est}}$',linewidth=0.8)
    plt.scatter(x1, v_est-1, color=colors[1], zorder=3, s=20)
    plt.scatter(x2, p, color=colors[0], zorder=3, s=20)
    plt.xlabel("Node index")
    plt.ylabel("Value")
    plt.title(r"Comparison of $\mathrm{p}$ and $\mathrm{p}_{\mathrm{est}}$")
    plt.ylim(0, 0.8)  # Set y-axis limits from 0 to 0.8
    plt.legend()
    plt.savefig(results_dir+f"/p_comparison_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/p_comparison_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")

    # Loss
    scale_factor = int(np.floor(np.log10(max(loss))))  # Get the order of magnitude
    normalized_loss = np.array(loss) / (10 ** scale_factor)  # Normalize loss
    plt.figure()
    plt.plot(normalized_loss, label=r"Loss", color="green", linestyle="--")
    plt.ylabel(f"Loss ($\\times 10^{{{scale_factor}}}$)")  # Adjust label to indicate scale
    plt.xlabel("Iteration")
    plt.title("Loss Convergence")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir+f"/loss_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.png")
    plt.savefig(results_dir+f"/loss_Normalized_m={m}_K={K}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")