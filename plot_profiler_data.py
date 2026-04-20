"""
Used to create graphs 

Example Run: 
python create_graphs.py --csv_file results.csv
"""

import csv
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="creates graphs"
)

parser.add_argument( 
    "--csv_file",
    help="sets the csv file"
)

args = parser.parse_args()


# function for graphing the change of a certain metric relative to batch size
def create_batch_power_graph(data_points, metric_key):
    print(f"evaluting {metric_key}")
    # a hashmap where the key is model-device and the value is a 2d array 
    # containing the batch size a metric being evaluated
    grouped_data = {}
    for data_point in data_points:
        print(data_point)
        key = data_point['device'] + "-" + data_point["model"]
        batch_size  = int(data_point["batch size"])
        metric = float(data_point[metric_key])
        print(f"\tkey: {key}")
        print(f"\tbatch_size: {batch_size}")
        print(f"\tmetric: {metric}")
        if(key in grouped_data):
            grouped_data[key]['batch_size'].append(batch_size) 
            grouped_data[key]['metric'].append(metric)
        else:
            grouped_data[key] = {}
            grouped_data[key]['batch_size'] = [batch_size]
            grouped_data[key]['metric'] = [metric]

    fig, ax = plt.subplots()
    ax.set_xscale("log", nonpositive='clip', base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel(metric_key)
    for key in grouped_data.keys(): 
        ax.plot(grouped_data[key]['batch_size'], grouped_data[key]['metric'], label=key)
    ax.legend()
    ax.set_title(f"Batch Size Vs {metric_key}")
    fig.savefig(f'graphs/batch_size vs {metric_key}.png')
    



def extract_data_points(csv_file_name):
    with open(csv_file_name) as f:
        return [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]


def main():
    csv_file_name = args.csv_file
    data_points = extract_data_points(csv_file_name)
    create_batch_power_graph(data_points, "time_to_first_token_sec")
    create_batch_power_graph(data_points, "tokens_per_second")
    create_batch_power_graph(data_points, "run_time_seconds")
    create_batch_power_graph(data_points, "total_energy_joules")
    create_batch_power_graph(data_points, "joules_per_token")
    create_batch_power_graph(data_points, "joules_per_token")
    # create_batch_power_graph(data_points, "FLOPS")





if __name__ == "__main__":
    main()


