"""
Used to create graphs 

Example Run: 
python create_graphs.py --csv_file results.csv
"""

import csv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="creates graphs"
)

parser.add_argument( 
    "--csv_file",
    help="sets the csv file"
)

args = parser.parse_args()


# function for graphing the change of a certain metric relative to batch size
def create_stacked_bar_chart(data_points, metric_keys):
    # a hashmap where the key is model-device and the value is a 2d array 
    # containing the batch size a metric being evaluated
    batch_sizes = []
    metrics = {}
    for metric_key in metric_keys:
        metrics[metric_key] = []
    for data_point in data_points:
        batch_size  = str(data_point["Batch Size"])
        batch_sizes.append(batch_size)
        for metric_key in metric_keys:
            metrics[metric_key].append(float(data_point[metric_key]))
    fig, ax = plt.subplots()
    width = 0.5
    bottom = np.zeros(len(data_points))
    for metric, values in metrics.items():
        p = ax.bar(batch_sizes, values, width, label=metric, bottom=bottom)
        bottom += values
        
    # ax.set_xscale("log", nonpositive='clip', base=2)
    # ax.set_xlabel("batch size")
    # ax.set_ylabel(metric_key)
    # for key in grouped_data.keys(): 
    #     ax.plot(grouped_data[key]['batch_size'], grouped_data[key]['metric'], label=key)
    ax.legend()
    ax.set_title(f"Batch Size Vs Runtime")
    fig.savefig(f'outputs/images/batch_size vs Runtime.png')



def extract_data_points(csv_file_name):
    with open(csv_file_name) as f:
        return [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]


def main():
    csv_file_name = args.csv_file
    data_points = extract_data_points(csv_file_name)
    create_stacked_bar_chart(data_points, ["Prefill Time (s)", "Decode Time (s)"])

if __name__ == "__main__":
    main()


