import matplotlib.pyplot as plt
import numpy as np


def create_boxplot(lev_distances_better, lev_distances_worse, lev_distances_eq,st_distances_better, st_distances_worse, st_distances_eq, ct):
    y = np.arange(len(lev_distances_worse) + len(lev_distances_better) + len(
        lev_distances_eq))  # np.random.randint(low=0, high=len(lev_distances_worse)+len(lev_distances_better)+len(lev_distances_eq), size=len(lev_distances_worse)+len(lev_distances_better)+len(lev_distances_eq))
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(lev_distances_better, positions=[1], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="lightblue", color="lightblue"))

    # Boxplot for mean_distance_eq
    bp2 = ax.boxplot(lev_distances_eq, positions=[2], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="tab:green", color="lightblue"))

    # Boxplot for mean_distance_worse
    bp3 = ax.boxplot(lev_distances_worse, positions=[3], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="tab:red", color="lightblue"))
    plt.rcParams["figure.figsize"] = (7, 5)

    plt.axhline(y=np.mean(st_distances_worse + st_distances_better + st_distances_eq), color='red',
                linestyle='--',
                label='Mean')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'D-DT Better ({len(st_distances_better)})', f'Equal ({len(st_distances_eq)})', f'D-DT Worse ({len(st_distances_worse)})'])
    plt.title('Hamming distances of 100 instances')
    plt.ylabel('Hamming distance')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'leven_scatter_ordered{ct}.png')
    plt.show()
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(st_distances_better, positions=[1], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="lightblue", color="lightblue"))

    # Boxplot for mean_distance_eq
    bp2 = ax.boxplot(st_distances_eq, positions=[2], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="tab:green", color="lightblue"))

    # Boxplot for mean_distance_worse
    bp3 = ax.boxplot(st_distances_worse, positions=[3], meanline=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor="tab:red", color="lightblue"))
    plt.rcParams["figure.figsize"] = (7, 5)

    plt.axhline(y=np.mean(st_distances_worse + st_distances_better + st_distances_eq), color='red',
                linestyle='--',
                label='Mean')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'D-DT Better ({len(st_distances_better)})', f'Equal ({len(st_distances_eq)})', f'D-DT Worse ({len(st_distances_worse)})'])

    plt.title('Start time distances of 100 instances')
    plt.ylabel('Start time distance')
    plt.legend([bp1['medians'][0], bp1['means'][0]], ['median', 'mean'])
    plt.tight_layout()
    plt.savefig(f'starttime_scatter_ordered{ct}.png')
    plt.show()
def plot_rtg_variations(values: dict):
    #Just much code for creating a plot for of
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    for ct, results in values.items():
        x, y = zip(*results["makespan"])
        plt.plot(x, y, label=f"K = {ct}")

    plt.title('Mean makespans of various rtgs on 100 instances')
    plt.ylabel('mean makespan')
    plt.xlabel('return-to-go factor')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'makespan.png')
    plt.show()

    for ct, results in values.items():
        x, y = zip(*results["Hamming"])
        plt.plot(x, y, label=f"K = {ct}")
    plt.title('Mean Hamming distance of various rtgs on 100 instances')
    plt.ylabel('mean Hamming distance')
    plt.xlabel('return-to-go factor')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'leven.png')
    plt.show()

    for ct, results in values.items():
        x, y = zip(*results["start_time"])
        plt.plot(x, y, label=f"K = {ct}")
    plt.title('Mean start time distance of various rtgs on 100 instances')
    plt.ylabel('mean start time distance')
    plt.xlabel('return-to-go factor')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'starttime.png')
    plt.show()