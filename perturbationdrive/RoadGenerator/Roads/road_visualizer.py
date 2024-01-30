import matplotlib.pyplot as plt
from typing import List, Tuple
from perturbationdrive.RoadGenerator.Roads.simulator_road import Road
import numpy as np


def visualize_road(road: Road, title="Road", plot_features=False, export_name=None):
    # get num turns
    num_turns = road.num_turns()
    curvature = road.curvature()
    waypoints = road.get_concrete_representation(to_plot=True)
    x = [point[0] for point in waypoints]
    y = [point[1] for point in waypoints]
    # make sure that x and y are the same length

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.plot(x, y, color="black", linewidth=10, linestyle="-")
    ax.plot(x, y, color="red", linewidth=2, linestyle=":")

    # Set the background color
    ax.set_facecolor("lightgreen")

    # Set the title
    ax.set_title(title, fontsize=10)
    # get range of x and y values
    diff_x = max(x) - min(x)
    diff_y = max(y) - min(y)
    if (diff_x) > (diff_y):
        # x is bigger
        axis_range = range(int(min(x)) - 1, int(max(x)) + 1)
        plt.xticks(axis_range)
        yaxis_range = range(
            int(np.mean(y) - diff_x * 0.5), int(np.mean(y) + diff_x * 0.5)
        )
        plt.yticks(yaxis_range)
    else:
        axis_range = range(int(min(y)) - 1, int(max(y)) + 1)
        plt.yticks(axis_range)
        xaxis_range = range(
            int(np.mean(x) - diff_y * 0.5), int(np.mean(x) + diff_y * 0.5)
        )
        plt.xticks(xaxis_range)

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if plot_features:
        ax.set_ylabel(f"Curvature: {curvature}", fontsize=20)
        ax.set_xlabel(f"Number of Curvers: {num_turns}", fontsize=20)

    # Remove the axes spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    if export_name is not None:
        plt.savefig(export_name, bbox_inches="tight")

    plt.show()
