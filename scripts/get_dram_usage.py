import csv
from auto_profiler import run_ncu_profile, extract_dataframe_from_ncu_files_via_csv, extract_dram_usage
import matplotlib.pyplot as plt
import numpy as np

batch_powers = [0, 5]
sequence_lengths = [100]
output_token_counts = [1]
model_names = ["google/gemma-2-2b", 'ridger/MMfreeLM-2.7B', "microsoft/phi-2"]

results = []

for model_name in model_names:
    for batch_power in batch_powers:
        for sequence_length in sequence_lengths:
            for output_token_count in output_token_counts:
                batch_size = 2**batch_power
                # run_ncu_profile(batch_size, output_token_count, sequence_length, model_name=model_name)
                df = extract_dataframe_from_ncu_files_via_csv(batch_size, output_token_count, sequence_length, model_name=model_name)
                dram_usage = extract_dram_usage(df)
                print(f"for {model_name}, at batch size: {batch_size} and sequence length {sequence_length}: {dram_usage} kbytes of dram were used")
                results.append({
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "output_token_count": output_token_count,
                    "dram_usage_kbytes": dram_usage,
                })

output_csv = "dram_usage_results.csv"
fieldnames = ["model_name", "batch_size", "sequence_length", "output_token_count", "dram_usage_kbytes"]

