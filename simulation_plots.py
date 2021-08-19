import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulation import simulation
##########
def plot_results(results):
    s_range = np.array(sorted(results['S'].unique()))
    l_range = np.array(sorted(results['l'].unique()))
    fig, axs = plt.subplots(1, 3, figsize = (12, 3))
    axs[0].violinplot(results.groupby('S')['MinAccuracy'].apply(lambda x: x.tolist()).to_list(), s_range)
    axs[0].set_xlabel("SoftMin Parameters", fontsize = 12)
    axs[0].set_ylabel("Minimum Estimate Accuracies", fontsize = 12)
    axs[0].set_title('Softmin Parameter Tuning', fontsize = 12)
    axs[0].set_xticks(s_range)
    axs[0].set_xticklabels(axs[0].get_xticks(), rotation = 340)

    axs[1].violinplot(results.groupby('l')['Accuracy'].apply(lambda x: x.tolist()).to_list(), l_range * 10)
    axs[1].set_xlabel("Regularization Parameters", fontsize = 12)
    axs[1].set_ylabel("Classification Accuracies", fontsize = 12)
    axs[1].set_title("Regularization Parameter Tuning", fontsize = 12)
    axs[1].set_xticks(np.round(l_range * 10, 2))
    axs[1].set_xticklabels(np.round(l_range, 2), rotation = 300)

    sc = axs[2].scatter(results['l'], results['S'], \
                        c = results['Accuracy'], vmin=min(results['Accuracy']), vmax=max(results['Accuracy']))
    axs[2].set_xlabel("Regularization Parameter", fontsize = 12)
    axs[2].set_ylabel('Softmin parameter', fontsize = 12)
    axs[2].set_title('Classification Accuracy vs. regularization and Softmin', fontsize = 12)
    axs[2].set_xticks(np.round(l_range, 1))
    axs[2].set_xticklabels(axs[2].get_xticks(), rotation = 300)
    axs[2].set_yticks(np.round(s_range, 2))
    axs[2].set_yticklabels(axs[2].get_yticks())
    
    fig.colorbar(sc)
    plt.tight_layout()
    plt.show()
    
def auto_plot_results(master):
    for k in master:
        print(k)
        plot_results(master[k])  


def plot_iterations(ax, df, method_name, x0, sim):
    l, S = df['l'].iloc[0], df['S'].iloc[0]
    if method_name == 'original':
        sim.optimize(l, S, x0=x0, scale = False, smooth_clip=True, lb = -np.inf, up = np.inf, intercept = 0, penalty_norm = False)
    if method_name == 'original_intercept':
        sim.optimize(l, S, x0=x0, scale = False, smooth_clip=True, lb = -np.inf, up = np.inf, intercept = 0.5, penalty_norm = False)
    if method_name == 'scaled':
        sim.optimize(l, S, x0=x0, scale = True, smooth_clip=True, lb = -np.inf, up = np.inf, intercept = 0, penalty_norm = False)
    if method_name == 'scaled_intercept':
        sim.optimize(l, S, x0=x0, scale = True, smooth_clip=True, lb = -np.inf, up = np.inf, intercept = 0.5, penalty_norm = False)
    if method_name == 'boundary_1':
        sim.optimize(l, S, x0=x0, scale = False, smooth_clip=False, lb = -1, up = 1, intercept = 0.5, penalty_norm = False)
    if method_name == 'boundary_5':
        sim.optimize(l, S, x0=x0, scale = False, smooth_clip=False, lb = -5, up = 5, intercept = 0.5, penalty_norm = False)
    if method_name == 'norm_penalty':
        sim.optimize(l, S, x0=x0, scale = False, smooth_clip=True, lb = -np.inf, up = np.inf, intercept = 0.5, penalty_norm = True)
    
    func_outputs = np.array([sim.objective_function(beta_iter, sim.X_dict_train, sim.y_dict, sim.total_rewards_dict, 
                                       sim.propensity_scores, sim.l, sim.S, sim.smooth_clip, None, None,
                                       sim.intercept, sim.penalty_norm)[0] for beta_iter in sim.beta_callback])
    X_dict_train_original = {k: sim.X_dict[k] for k in sim.X_dict_train}
    if sim.intercept:
        VF_outputs = np.array([sim.VF(X_dict_train_original, sim.X_dict_train, lambda x, pid, idx: np.dot(betas, np.hstack([1, x]))) \
                               for betas in sim.beta_callback])
    else:
        VF_outputs = np.array([sim.VF(X_dict_train_original, sim.X_dict_train, lambda x, pid, idx: np.dot(betas, x)) \
                               for betas in sim.beta_callback]) 
    pelnalties = np.array([sim.l * np.sum(beta**2) for beta in sim.beta_callback])
    ax.plot(np.arange(func_outputs.shape[0]), func_outputs, label = 'Objective Function', marker = 'o')
    ax.plot(np.arange(VF_outputs.shape[0]), VF_outputs, label = 'Value Function', marker = 'o')
    ax.plot(np.arange(VF_outputs.shape[0]), pelnalties, label = 'penalty', marker = 'o')
    ax.plot(np.arange(VF_outputs.shape[0]), func_outputs - pelnalties, label = 'loss', marker = 'o')
    ax.set_xlabel("Iteration number", fontsize = 12)
    ax.set_title(method_name, fontsize = 15)
    
    
def auto_plot_iterations(master, x0, sim, *x_distr_args):
    fig, axes = plt.subplots(2, 4, figsize = (20, 8))
    for i, k in enumerate(master.keys()):
        print(k)
        best_df = master[k].loc[master[k]['Accuracy'].idxmax(), ].to_frame().transpose()
        display(best_df)
        sim.reset_data(sim.n, sim.x_distr)
        sim.generate_data(*x_distr_args)
        plot_iterations(axes[i//4, i%4], best_df, k, x0, sim)
    handles, labels = axes[1, -2].get_legend_handles_labels()
    axes[-1, -1].legend(handles, labels, loc='center', fontsize = 18)
    plt.tight_layout()
    plt.show()

    