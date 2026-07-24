import csv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from matplotlib.lines import Line2D   # only needed for a custom legend (optional)

GPU_MATMUL_FILE = "paper_csv/mmfree.csv"
GPU_BITNET_FILE = "paper_csv/bitnet.csv"
FPGA_FILE = "paper_csv/fpga.csv"

GPU_SCALED_MATMUL_FILE = "paper_csv/scaled_mmfree.csv"
GPU_SCALED_BITNET_FILE = "paper_csv/scaled_bitnet.csv"

TPS_COL = "tokens_per_second"
MODEL_COL = "model"
BS_COL = "batch size"
PACKING_COL = "Weight Packing"
DEVICE_COL = "device"
TTFT_COL = "Avg Prefill Time (s)"

RIDGER_MODEL_ID = "ridger/MMfreeLM-2.7B"
BITNET_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
SCALED_MMFREE_MODEL_ID = "100B MatMulFreeLM"
SCALED_BITNET_MODEL_ID = "Scaled Up Bitnet"

V100_DEVICE_ID = "Tesla V100-SXM2-32GB"
MULTI_V100_DEVICE_ID = "Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB,Tesla V100-SXM2-32GB"
A40_DEVICE_ID = "NVIDIA A40"
U250_DEVICE_ID = "U250"


def get_col_from_df(df, hardware, model, val_col, packing=None): 
    df = df[(df[MODEL_COL] == model) & (df[DEVICE_COL] == hardware)]
    if packing is not None: 
        df = df[df[PACKING_COL] == packing]
    return df[val_col]

def get_value_from_df(df, hardware, model, find_max, max_col, val_col, packing=None): 
    df = df[(df[MODEL_COL] == model) & (df[DEVICE_COL] == hardware)]
    if packing is not None: 
        df = df[df[PACKING_COL] == packing]
    return df.loc[df[max_col].idxmax()][val_col] if find_max else df.loc[df[max_col].idxmin()][val_col]
    
def performance_barchart(matmul_gpu_df, bitnet_gpu_df, fpga_df):

    best_v100_matmul_packing_tps = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, TPS_COL, packing=True)
    best_v100_matmul_packing_bs = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, BS_COL, packing=True)

    best_v100_matmul_no_packing_tps = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, TPS_COL, packing=False)
    best_v100_matmul_no_packing_bs = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, BS_COL, packing=False)

    best_U250_matmul_tps = get_value_from_df(fpga_df, U250_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, TPS_COL)
    best_U250_matmul_bs = get_value_from_df(fpga_df, U250_DEVICE_ID, RIDGER_MODEL_ID, True, TPS_COL, BS_COL)

    best_v100_bitnet_packing_tps = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, TPS_COL, packing=True)
    best_v100_bitnet_packing_bs = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, BS_COL, packing=True)

    best_v100_bitnet_no_packing_tps = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, TPS_COL, packing=False)
    best_v100_bitnet_no_packing_bs = get_value_from_df(matmul_gpu_df, V100_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, BS_COL, packing=False)

    best_a40_bitnet_packing_tps = get_value_from_df(bitnet_gpu_df, A40_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, TPS_COL, packing=True)
    best_a40_bitnet_packing_bs = get_value_from_df(bitnet_gpu_df, A40_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, BS_COL, packing=True)

    best_a40_bitnet_no_packing_tps = get_value_from_df(bitnet_gpu_df, A40_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, TPS_COL, packing=False)
    best_a40_bitnet_no_packing_bs = get_value_from_df(bitnet_gpu_df, A40_DEVICE_ID, BITNET_MODEL_ID, True, TPS_COL, BS_COL, packing=False)

    group_names = ['MatMulFree', 'Bitnet']
    group_idx   = np.arange(len(group_names))
    
    group_series = {
        0: [
            ('V100 Weight Packing', best_v100_matmul_packing_tps, f'{best_v100_matmul_packing_bs}'),
            ('V100 Non-Weight Packing', best_v100_matmul_no_packing_tps, f'{best_v100_matmul_no_packing_bs}'), 
            ('U250', best_U250_matmul_tps, f'{best_U250_matmul_bs}')
        ],
        1: [
            ('V100 Weight Packing', best_v100_bitnet_packing_tps, f'{best_v100_bitnet_packing_bs}'),
            ('V100 Non-Weight Packing', best_v100_bitnet_no_packing_tps, f'{best_v100_bitnet_no_packing_bs}'),
            ('A40 Weight Packing', best_a40_bitnet_packing_tps, f'{best_a40_bitnet_packing_bs}'),
            ('A40 Non-Weight Packing', best_a40_bitnet_no_packing_tps, f'{best_a40_bitnet_no_packing_bs}'),
        ]
    }
    all_labels = sorted({lbl for lst in group_series.values() for lbl, *_ in lst})
    cmap = plt.cm.tab10
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(all_labels)}

    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(7, 5))

    added_to_legend = set()

    for g_idx, series in group_series.items():
        n = len(series)
        start_offset = - (n - 1) / 2 * bar_width

        for i, (lbl, height, unit) in enumerate(series):
            x = g_idx + start_offset + i * bar_width

            rect = ax.bar(x,
                        height,
                        width=bar_width,
                        color=label_to_color[lbl],
                        edgecolor='black',
                        label=lbl if lbl not in added_to_legend else None)
            added_to_legend.add(lbl)

            label_text = f'{unit}'
            ax.bar_label(rect, labels=[label_text],
                        fontsize=9,
                        padding=3,
                        color='black')

    ax.set_xlabel('Model')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Tokens per Second vs Model & Hardware Configuration')
    ax.set_xticks(group_idx)
    ax.set_xticklabels(group_names)

    ax.set_xlim(-0.5, len(group_names) - 0.5)

    max_h = max(
        best_v100_matmul_packing_tps,
        best_v100_matmul_no_packing_tps,
        best_v100_bitnet_packing_tps,
        best_v100_bitnet_no_packing_tps,
        best_a40_bitnet_packing_tps,
        best_a40_bitnet_no_packing_tps,
    )
    ax.set_ylim(0, max_h * 1.05)  

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True,ncol=2)


    ax.grid(axis='y', linestyle='--', alpha=0.6)

    out_path = 'paper_graphs/normal_models_tps.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'✅ Figure saved to {out_path}')

def scaled_performance_bar_chart(scaled_matmul_gpu_df, scaled_bitnet_gpu_df):

    best_multi_gpu_v100_matmul_tps = get_value_from_df(scaled_matmul_gpu_df, MULTI_V100_DEVICE_ID, SCALED_MMFREE_MODEL_ID, True, TPS_COL, TPS_COL)
    best_multi_gpu_v100_matmul_bs = get_value_from_df(scaled_matmul_gpu_df, MULTI_V100_DEVICE_ID, SCALED_MMFREE_MODEL_ID, True, TPS_COL, BS_COL)

    best_single_gpu_v100_matmul_tps = get_value_from_df(scaled_matmul_gpu_df, V100_DEVICE_ID, SCALED_MMFREE_MODEL_ID, True, TPS_COL, TPS_COL)
    best_single_gpu_v100_matmul_bs = get_value_from_df(scaled_matmul_gpu_df, V100_DEVICE_ID, SCALED_MMFREE_MODEL_ID, True, TPS_COL, BS_COL)

    best_single_gpu_a40_bitnet_tps = get_value_from_df(scaled_bitnet_gpu_df, A40_DEVICE_ID, SCALED_BITNET_MODEL_ID, True, TPS_COL, TPS_COL)
    best_single_gpu_a40_bitnet_bs = get_value_from_df(scaled_bitnet_gpu_df, A40_DEVICE_ID, SCALED_BITNET_MODEL_ID, True, TPS_COL, BS_COL)

    group_names = ['100B MatMulFree', '100B Bitnet']
    group_idx   = np.arange(len(group_names))
    
    group_series = {
        0: [
            ('8xV100 No Weight Packing', best_multi_gpu_v100_matmul_tps, f'{best_multi_gpu_v100_matmul_bs}'),
            ('V100 Weight Packing', best_single_gpu_v100_matmul_tps, f'{best_single_gpu_v100_matmul_bs}')
        ],
        1: [
            ('A40', best_single_gpu_a40_bitnet_tps, f'{best_single_gpu_a40_bitnet_bs}')
        ]
    }
    all_labels = sorted({lbl for lst in group_series.values() for lbl, *_ in lst})
    cmap = plt.cm.tab10
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(all_labels)}

    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(7, 5))

    added_to_legend = set()

    for g_idx, series in group_series.items():
        n = len(series)
        start_offset = - (n - 1) / 2 * bar_width

        for i, (lbl, height, unit) in enumerate(series):
            x = g_idx + start_offset + i * bar_width

            rect = ax.bar(x,
                        height,
                        width=bar_width,
                        color=label_to_color[lbl],
                        edgecolor='black',
                        label=lbl if lbl not in added_to_legend else None)
            added_to_legend.add(lbl)

            label_text = f'{unit}'
            ax.bar_label(rect, labels=[label_text],
                        fontsize=9,
                        padding=3,
                        color='black')

    ax.set_xlabel('Model')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Tokens per Second vs Model & Hardware Configuration')
    ax.set_xticks(group_idx)
    ax.set_xticklabels(group_names)

    ax.set_xlim(-0.5, len(group_names) - 0.5)

    max_h = max(
        best_multi_gpu_v100_matmul_tps,
        best_single_gpu_v100_matmul_tps,
        best_single_gpu_a40_bitnet_tps,
    )
    ax.set_ylim(0, max_h * 1.05)  

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True,ncol=2)


    ax.grid(axis='y', linestyle='--', alpha=0.6)

    out_path = 'paper_graphs/scaled_models_tps.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'✅ Scaled Performance Graphs saved to {out_path}')

def latency_requirements(matmul_gpu_df, fpga_df): 
    v100_packing_ttft = get_col_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, TTFT_COL, packing=True)
    v100_packing_tps = get_col_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, TPS_COL, packing=True)

    v100_no_packing_ttft = get_col_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, TTFT_COL, packing=False)
    v100_no_packing_tps = get_col_from_df(matmul_gpu_df, V100_DEVICE_ID, RIDGER_MODEL_ID, TPS_COL, packing=False)

    u250_ttft = get_col_from_df(fpga_df, U250_DEVICE_ID, RIDGER_MODEL_ID, TTFT_COL)
    u250_tps = get_col_from_df(fpga_df, U250_DEVICE_ID, RIDGER_MODEL_ID, TPS_COL)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(v100_packing_tps, v100_packing_ttft, label = "V100 Weight Packing")
    ax.plot(v100_no_packing_tps, v100_no_packing_ttft, label = "V100 No Weight Packing")
    ax.plot(u250_tps, u250_ttft, label = "U250")

    ax.set_xlabel('Tokens per Second')
    ax.set_ylabel('Avg Prefill Time (s)')
    ax.set_title('Throughput Vs Latency: TPS vs TTFT')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    out_path="paper_graphs/latency_graph.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'✅ Latency graph saved to {out_path}')

def main(): 
    matmul_gpu_df = pd.read_csv(GPU_MATMUL_FILE)
    bitnet_gpu_df = pd.read_csv(GPU_BITNET_FILE)
    fpga_df = pd.read_csv(FPGA_FILE)

    scaled_matmul_gpu_df = pd.read_csv(GPU_SCALED_MATMUL_FILE)
    scaled_bitnet_gpu_df = pd.read_csv(GPU_SCALED_BITNET_FILE)


    performance_barchart(matmul_gpu_df, bitnet_gpu_df,fpga_df)
    latency_requirements(matmul_gpu_df, fpga_df)
    scaled_performance_bar_chart(scaled_matmul_gpu_df, scaled_bitnet_gpu_df)
if __name__ == "__main__":
    main()
