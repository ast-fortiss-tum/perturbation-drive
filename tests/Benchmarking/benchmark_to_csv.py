import argparse
import csv
import pyperf
import statistics


def export_csv(args, bench):
    runs = bench.get_runs()
    runs_values = [run.values for run in runs if run.values]

    rows = []
    for run_values in zip(*runs_values):
        mean = statistics.mean(run_values)
        st_dev = statistics.stdev(run_values)
        rows.append([mean, st_dev])

    with open(args.csv_filename, "a", newline="", encoding="ascii") as fp:
        writer = csv.writer(fp)
        # Write benchmark name and then the mean values
        writer.writerow([bench.get_name()] + rows[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_filename")
    parser.add_argument("csv_filename")
    return parser.parse_args()


def main():
    """
    Extracts mean time and standard deviation of all benchmarks into a csv file

    Use python3 tests/Benchmarking/benchmark_to_csv.py benchmark-pert.json benchmark_perturbation.csv
    """
    args = parse_args()

    # Load the benchmark suite
    suite = pyperf.BenchmarkSuite.load(args.json_filename)

    # Open (or create) the CSV file and write the header
    with open(args.csv_filename, "w", newline="", encoding="ascii") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Benchmark", "Mean", "st_dev"])

    # Iterate over all benchmarks and export them to the CSV
    for bench in suite:
        export_csv(args, bench)


if __name__ == "__main__":
    main()
