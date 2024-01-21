import matplotlib.pyplot as plt
from typing import List, Tuple
from perturbationdrive.RoadGenerator.Roads.simulator_road import Road


def visualize_road(road: Road, title: str = "Road"):
    waypoints = road.get_concrete_representation(to_plot=True)
    x = [point[0] for point in waypoints]
    y = [point[1] for point in waypoints]
    # make sure that x and y are the same length

    plt.plot(
        x,
        y,
        color="black",
        linewidth=2,
        linestyle="-",
    )
    ax = plt.axes()
    # make the background of the plot light green
    ax.grid(False)
    ax.set_facecolor("green")
    ax.set_yticklabels([])

    # get range of x and y values
    if (max(x) - min(x)) > (max(y) - min(y)):
        # x is bigger
        axis_range = range(int(min(x)), int(max(x)))
    else:
        axis_range = range(int(min(y)), int(max(y)))
    plt.xticks(axis_range)
    plt.yticks(axis_range)
    # Remove tick labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"{title}: Turns={road.compute_num_turns()[0]}, Smoothness={road.calculate_smoothness()}",
        fontsize=20,
    )

    # Remove the axes spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # add a title to the plot
    plt.show()
