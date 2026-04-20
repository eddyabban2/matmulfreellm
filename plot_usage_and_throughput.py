import csv
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # <-- NEW: Necessary for vectorized operations

# --- Function Definitions ---

def extend_metric_for_cycles(df: pd.DataFrame, metric: str) -> list:
    """
    Extends a metric for the number of cycles that it occurs.
    Replaces the slow row-by-row iteration with fast numpy vectorization.
    """
    
    # 1. Get the cycle counts (must be integers)
    cycles_column = "sm__cycles_elapsed.avg (cycle)"
    
    # Ensure the column is numeric and cast to int (the number of repetitions)
    cycle_counts = df[cycles_column].astype(float).astype(int).values 
    
    # 2. Get the metric values (what we want to repeat)
    # Ensure the metric column is float
    values_to_repeat = df[metric].astype(float).values
    
    # 3. Use np.repeat to perform the entire transformation in one vectorized step
    # This is significantly faster than looping.
    return np.repeat(values_to_repeat, cycle_counts).tolist()


def plot_metric(df: pd.DataFrame, metric: str):
    ends = df["sm__cycles_active.avg (cycle)"].cumsum().to_numpy()
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    x_path = np.empty(len(df) * 2)
    x_path[0::2] = starts
    x_path[1::2] = ends

    y_vals = df[metric].to_numpy()
    y_path = np.repeat(y_vals, 2)

    ax.plot(x_path, y_path, linewidth=2)
    # ax.set_xlabel("Cycles")
    # ax.set_ylabel(metric)
    # ax.set_title(f"{metric} over Cycles")
    # plt.tight_layout()
    # plt.show()
    
    ax.set_xlabel("Time Step (Cycles)")
    ax.set_ylabel(metric)
    ax.set_title(f"Usage Plot for {metric}")
    ax.grid(True)
    fig.savefig(f'outputs/images/Cycles vs {metric}.png')

# --- Main Execution ---

def main():
    """Reads data and generates plots."""
    parser = argparse.ArgumentParser(
        description="creates plots showing usage and throughput data"
    )
    parser.add_argument(
        "--csv",
        help="sets the csv file name",
        required=True # Added required=True for robustness
    )
    
    args = parser.parse_args()
    
    # 1. Import data
    csv_name = args.csv
    
    # Optimization Tip: If you know the data types, pass them to pd.read_csv
    # This can prevent Pandas from having to infer types row-by-row.
    # df = pd.read_csv(csv_name, dtype={'column_name': np.float32, 'integer_col': np.int32})
    try:
        df = pd.read_csv(csv_name)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_name}")
        return
    print(f"Kernel Count: {len(df)}")
    df["fraction of active cycles"] = df["sm__cycles_active.avg (cycle)"] / df["sm__cycles_elapsed.avg (cycle)"]
    # check to see if 
    # 2. Plot the required metrics
    plot_metric(df, "dram__throughput.sum.pct_of_peak_sustained_elapsed (%)")
    plot_metric(df, "sm__throughput.avg.pct_of_peak_sustained_elapsed (%)")
    plot_metric(df, "Compute Intensity")
    plot_metric(df, "fraction of active cycles")
    
    

if __name__ == "__main__":
    main()