import csv
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib.cm as cm

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
    ends = df["sm__cycles_elapsed.avg (cycle)"].cumsum().to_numpy()
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
    fig.savefig(f'outputs/images/Cycles vs {metric.replace("/", " per ")}.png')

def plot_multi_metric(df: pd.DataFrame, metrics: list, ylabel: str, title: str):
    ends = df["sm__cycles_elapsed.avg (cycle)"].cumsum().to_numpy()
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    for metric in metrics:
        x_path = np.empty(len(df) * 2)
        x_path[0::2] = starts
        x_path[1::2] = ends
        y_vals = df[metric].to_numpy()
        y_path = np.repeat(y_vals, 2)
        ax.plot(x_path, y_path, linewidth=2, label=metric)
    start_point = 0 
    end_point = 0
    seen_labels = set()

    for row in df.to_dict(orient="records"):
        nvtx_col = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
        start_point = end_point
        end_point = start_point + float(row["sm__cycles_elapsed.avg (cycle)"])

        spans = []
        if 'HGRNBitAttentionForward' in row[nvtx_col]:
            spans.append(('Attention Kernels', 'blue', None))
        if 'HGRNBitMLP' in row[nvtx_col]:
            spans.append(('HGRN MLP Kernels', 'yellow', None))
        if 'linearFunction' in row[nvtx_col]:
            spans.append(('Ternary Matmul Kernels ', 'white', "///"))

        for label, color, hatch in spans:
            # Only pass label the first time; use '_' prefix to suppress duplicates
            legend_label = label if label not in seen_labels else f"_{label}"
            if hatch is None:
                plt.axvspan(start_point, end_point, label=legend_label, color=color, alpha=0.15)
            else: 
                plt.axvspan(start_point, end_point, label=legend_label, alpha=0.5, hatch=hatch, facecolor='none', edgecolor="green")
            seen_labels.add(label)


    ax.set_xlabel("Time Step (Cycles)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)
    ax.grid(True)
    fig.savefig(f'outputs/images/{title}.png', bbox_inches='tight')

def plot_multi_metric_stacked_bar_chart(df, ylabel, title):
    stall_cols  = [c for c in df.columns if 'pcsamp_warps_issue_stalled' in c]
    short_names = [c.replace('smsp__pcsamp_warps_issue_stalled_', '').split(' ')[0]
                   for c in stall_cols]

    durations  = df['gpu__time_duration.sum (us)'].values
    total_width = len(df)
    bar_widths  = (durations / durations.sum()) * total_width
    left_edges  = np.concatenate([[0], np.cumsum(bar_widths)[:-1]])
    x_centers   = left_edges + bar_widths / 2

    # Build a dense x-grid so fill_between draws smooth continuous bands
    x_dense = np.concatenate([
        np.linspace(left_edges[i], left_edges[i] + bar_widths[i], 50)
        for i in range(len(df))
    ])

    colors = cm.tab20(np.linspace(0, 1, len(stall_cols)))

    fig, ax = plt.subplots(figsize=(18, 6))

    # Interpolate each stall metric onto the dense x-grid
    cumulative = np.zeros(len(x_dense))
    for col, color in zip(stall_cols, colors):
        # Step-interpolate: each kernel's value holds constant across its bar width
        values_dense = np.concatenate([
            np.full(50, row[col]) for _, row in df.iterrows()
        ])
        ax.fill_between(x_dense, cumulative, cumulative + values_dense,
                        color=color, alpha=0.85, linewidth=0)
        # Thin boundary line to separate layers
        ax.plot(x_dense, cumulative + values_dense,
                color=color, linewidth=0.4, alpha=0.6)
        cumulative += values_dense

    ax.set_xlim(0, total_width)
    ax.set_xticks(x_centers)
    shorted_kernel_names = df['Kernel Name'].tolist()
    shorted_kernel_names = ["Ternary MatMul" if "gemm" in kernel_name else kernel_name for kernel_name in shorted_kernel_names]
    shorted_kernel_names = ["Mul Functor" if "binary_internal::MulFunctor<float>>>" in kernel_name else kernel_name for kernel_name in shorted_kernel_names]
    shorted_kernel_names = ["Add Functor" if "at::CUDAFunctorOnOther_add<c10::Half>" in kernel_name else kernel_name for kernel_name in shorted_kernel_names]
    shorted_kernel_names = ["Sigmoid" if "at::sigmoid_kernel_cuda" in kernel_name else kernel_name for kernel_name in shorted_kernel_names]
    ax.set_xticklabels(shorted_kernel_names, rotation=45, ha='right', fontsize=7)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Kernel (width ∝ duration in µs)')
    ax.set_title(title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(
        handles[::-1], short_names[::-1],
        title='Stall Type',
        title_fontsize=9,
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=8,
        framealpha=0.9,
        edgecolor='gray',
        borderpad=0.8,
    )

    fig.savefig('outputs/images/stall_area_chart.png', bbox_inches='tight')
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
    df["fraction of active cycles"] = df["smsp__cycles_active.avg (cycle)"] / df["smsp__cycles_elapsed.sum (cycle)"]
    # df["Half Preciscion Instruction Count"] = 
    # check to see if 
    # 2. Plot the required metrics
    plot_metric(df, "dram__throughput.sum.pct_of_peak_sustained_elapsed (%)")
    plot_metric(df, "sm__throughput.avg.pct_of_peak_sustained_elapsed (%)")
    plot_metric(df, "Compute Intensity")
    plot_metric(df, "fraction of active cycles")
    plot_metric(df, "dram__throughput.sum.pct_of_peak_sustained_elapsed (%)")
    # plot_metric(df, "Single Precision GFLOP/s")
    plot_metric(df, "Half Precision GFLOP/s")
    plot_metric(df, "Double Precision GFLOP/s")
    plot_multi_metric(df, [
        "Single Precision GFLOP/s", 
        "Half Precision GFLOP/s", 
        "Half Precision Matrix Multiply and Accumulate Instructions (Billion Inst/s)"], "GFLOP Per Second", "~GFLOPs Per Second")
    
    plot_multi_metric(df, ["Single Precision GFLOP/s"], "Single Precision GFLOP Per Second", "Single Precision GFLOPs Per Second")
    plot_multi_metric(df, ["dram__throughput.sum.pct_of_peak_sustained_elapsed (%)", "sm__throughput.avg.pct_of_peak_sustained_elapsed (%)"], "DRAM and Streaming Multiprocessor Throughput", "% of theortical peak sustained")
    plot_multi_metric_stacked_bar_chart(df, "stall count", "Stall Count")
    
    

if __name__ == "__main__":
    main()