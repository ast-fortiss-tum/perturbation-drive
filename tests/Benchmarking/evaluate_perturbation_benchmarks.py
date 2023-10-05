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
    error = [data["st_dev"], data["st_dev"]]
    ax = sns.barplot(x="Benchmark", y="Mean", data=data, capsize=0.1, errorbar=None)
    # Adding error bars. Note that this can be tricky when using `hue` in seaborn as seaborn does not directly support it.
    # for i, benchmark in enumerate(data['Benchmark']):
    #    plt.errorbar(x=i, y=data['Mean'].iloc[i], yerr=data['st_dev'].iloc[i], color='black', capsize=3)
    plt.errorbar(data["Benchmark"], y=data["Mean"], yerr=error, fmt="none", c="r")
    plt.xticks(rotation=90)
    plt.title("Mean and StdDev for Each Benchmark Separately")
    plt.ylabel("Mean Time in s")
    plt.xlabel("Benchmark")
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
    ax2 = sns.barplot(x=grouped_data.index, y="Mean", data=grouped_data, capsize=0.1)
    plt.errorbar(
        grouped_data.index, y=grouped_data["Mean"], yerr=error, fmt="none", c="r"
    )
    plt.xticks(rotation=90)
    plt.title("Mean and Aggregated StdDev for Benchmark over all scales")
    plt.ylabel("Mean Time in s")
    plt.xlabel("Benchmark Prefix")
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
    plot_benchmarks(args)


if __name__ == "__main__":
    main()
