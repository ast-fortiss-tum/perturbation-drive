import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import matplotlib


def plot_benchmarks(args: argparse.Namespace):
    # Load the data
    data = pd.read_csv(args.csv_filename)

    # Extract prefix (assuming it's the substring before the first space)
    data["Prefix"] = data["Benchmark"].apply(lambda x: x.split(" ")[0])
    data = data.sort_values(by="Prefix")

    # Plot 1: All benchmarks separately
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    # add a red line at y=50
    plt.axvline(x=50, color="r", linestyle="-")
    error = [data["st_dev"], data["st_dev"]]
    ax = sns.barplot(x="Mean", y="Benchmark", data=data, capsize=0.1, errorbar=None)
    # Adding error bars. Note that this can be tricky when using `hue` in seaborn as seaborn does not directly support it.
    # for i, benchmark in enumerate(data['Benchmark']):
    #    plt.errorbar(x=i, y=data['Mean'].iloc[i], yerr=data['st_dev'].iloc[i], color='black', capsize=3)
    plt.errorbar(x=data["Mean"], y=data["Benchmark"], yerr=error, fmt="none", c="r")
    plt.xticks(rotation=90)
    plt.title("Mean and StdDev for Each Benchmark Separately")
    plt.xlabel("Mean Time in s")
    plt.ylabel("Benchmark")
    plt.tight_layout()
    plt.show()

    # Plot 2: Benchmarks grouped by prefix
    grouped_data = data.groupby("Prefix").agg(
        {
            "Mean": "mean",
            # For simplicity, using sum of std devs as a measure of total variation
            "st_dev": "sum",
        }
    )
    error = [grouped_data["st_dev"], grouped_data["st_dev"]]
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x="Mean", y=grouped_data.index, data=grouped_data, capsize=0.1)
    plt.errorbar(
        x=grouped_data["Mean"], y=grouped_data.index, yerr=error, fmt="none", c="r"
    )
    plt.xticks(rotation=90)
    plt.title("Mean and StdDev for all Perturbations")
    plt.xlabel("Mean Time in s")
    plt.ylabel("Benchmark Prefix")
    plt.tight_layout()
    plt.show()


def plot_benchmarks_reverse(args: argparse.Namespace, exclude_prefixes: list = []):
    # Load the data
    data = pd.read_csv(args.csv_filename)

    # filter rows based on prefix
    data = data[~data["Benchmark"].str.startswith(tuple(exclude_prefixes))]

    # Convert "Mean" column to float
    data["Mean"] = data["Mean"].astype(float)

    # cast mean and st_dev from seconds to milliseoncs
    data["Mean"] = data["Mean"] * 1000
    data["st_dev"] = data["st_dev"] * 1000

    # take all but the last two chars
    data["Prefix"] = data["Benchmark"].apply(lambda x: x[:-2])
    # replace _ with space
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("_", " "))
    data = data.sort_values(by="Prefix")

    # replace scale by intensity
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("scale", "intensity"))
    data["Benchmark"] = data["Benchmark"].apply(
        lambda x: x.replace("scale", "intensity")
    )

    # Plot 1: All benchmarks separately
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(25, 12))

    for i in range(5):
        plt.figure(figsize=(25, 12))  # create a new figure for each plot

        # filter for all perturbations on scale i
        data_i = data[data["Benchmark"].str.endswith(str(i))]
        error = [data_i["st_dev"], data_i["st_dev"]]
        ax = sns.barplot(
            x="Benchmark", y="Mean", data=data_i, capsize=0.1, errorbar=None
        )
        # add red line at x=50
        ax.axhline(y=50, color="r", linestyle="-")

        plt.errorbar(
            data_i["Benchmark"], y=data_i["Mean"], yerr=error, fmt="none", c="r"
        )
        plt.xticks(
            rotation=45,
            horizontalalignment="right",
            fontweight="light",
            fontsize="x-large",
        )
        plt.title("Mean and StdDev")
        plt.ylabel("Mean Time in ms")
        plt.xlabel("Perturbation on intensity " + str(i))
        plt.tight_layout()
        plt.show()

    # Plot 2: Benchmarks grouped by prefix
    # remove the word "scale" from the prefix
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("intensity", ""))
    # remove leading and trailing white spaces
    data["Prefix"] = data["Prefix"].apply(lambda x: x.strip())
    grouped_data = data.groupby("Prefix").agg(
        {
            "Mean": "mean",
            "st_dev": "mean",
        }
    )
    error = [grouped_data["st_dev"], grouped_data["st_dev"]]
    plt.figure(figsize=(25, 12))
    ax2 = sns.barplot(x=grouped_data.index, y="Mean", data=grouped_data, capsize=0.1)
    ax2.axhline(y=50, color="r", linestyle="-")

    plt.errorbar(
        grouped_data.index,
        y=grouped_data["Mean"].astype(float),
        yerr=error,
        fmt="none",
        c="r",
    )
    plt.xticks(
        rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
    )
    plt.title("Mean and Aggregated StdDev for Benchmark over all intensities")
    plt.ylabel("Mean Time in ms")
    plt.xlabel("Perturbation")
    plt.tight_layout()
    plt.show()


def compare_attention_perturbation(args: argparse.Namespace):
    """
    Compare the computational overhead of the attention perturbation
    """
    # Load the data
    data = pd.read_csv(args.csv_filename)

    # only keep rows with "gaussian_noise" in the Benchmark column
    data = data[data["Benchmark"].str.contains("gaussian_noise")]

    # Convert "Mean" column to float
    data["Mean"] = data["Mean"].astype(float)

    # cast mean and st_dev from seconds to milliseoncs
    data["Mean"] = data["Mean"] * 1000
    data["st_dev"] = data["st_dev"] * 1000

    # take all but the last two chars
    data["Prefix"] = data["Benchmark"].apply(lambda x: x[:-2])
    # replace _ with space
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("_", " "))
    data = data.sort_values(by="Prefix")

    # replace scale by intensity
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("scale", "intensity"))
    data["Benchmark"] = data["Benchmark"].apply(
        lambda x: x.replace("scale", "intensity")
    )

    # Plot 1: All benchmarks separately
    sns.set_theme(style="whitegrid")

    for i in range(5):
        plt.figure(figsize=(25, 12))  # create a new figure for each plot

        # filter for all perturbations on scale i
        data_i = data[data["Benchmark"].str.endswith(str(i))]
        error = [data_i["st_dev"], data_i["st_dev"]]
        ax = sns.barplot(
            x="Benchmark", y="Mean", data=data_i, capsize=0.1, errorbar=None
        )
        # add red line at x=50
        ax.axhline(y=50, color="r", linestyle="-")
        ax.bar_label(ax.containers[0])
        plt.errorbar(
            data_i["Benchmark"], y=data_i["Mean"], yerr=error, fmt="none", c="r"
        )
        plt.xticks(
            rotation=45,
            horizontalalignment="right",
            fontweight="light",
            fontsize="x-large",
        )
        plt.title("Mean and StdDev")
        plt.ylabel("Mean Time in ms")
        plt.xlabel("Perturbation on intensity " + str(i))
        plt.tight_layout()
        plt.show()

    # Plot 2: Benchmarks grouped by prefix
    # remove the word "scale" from the prefix
    data["Prefix"] = data["Prefix"].apply(lambda x: x.replace("intensity", ""))
    # remove leading and trailing white spaces
    data["Prefix"] = data["Prefix"].apply(lambda x: x.strip())
    grouped_data = data.groupby("Prefix").agg(
        {
            "Mean": "mean",
            "st_dev": "mean",
        }
    )
    error = [grouped_data["st_dev"], grouped_data["st_dev"]]
    plt.figure(figsize=(25, 12))
    ax2 = sns.barplot(x=grouped_data.index, y="Mean", data=grouped_data, capsize=0.1)
    ax2.axhline(y=50, color="r", linestyle="-")

    plt.errorbar(
        grouped_data.index,
        y=grouped_data["Mean"].astype(float),
        yerr=error,
        fmt="none",
        c="r",
    )
    ax2.bar_label(ax2.containers[0])
    plt.xticks(
        rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
    )
    plt.title("Mean and Aggregated StdDev for Gaussian Noise with and without Attention-based Perturbation")
    plt.ylabel("Mean Time in ms")
    plt.xlabel("Perturbation")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filename")
    return parser.parse_args()


def main():
    """
    Plots the mean time and standard deviation of all benchmarks of a csv file

    Use python3 tests/Benchmarking/evaluate_perturbation_benchmarks.py benchmark_perturbation.csv
    """
    args = parse_args()
    plot_benchmarks_reverse(args)


if __name__ == "__main__":
    main()
