#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate mmfree
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 0 --max_batch_power 0 2>&1 | tee log1.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 0 --max_batch_power 0 --compression 2>&1 | tee log2.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 8 --max_batch_power 8 2>&1 | tee log3.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 8 --max_batch_power 8 --compression 2>&1 | tee log4.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 0 --max_batch_power 0 --model_name scaled_mmfree 2>&1 | tee log9.log
python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 6 --max_batch_power 6 --model_name scaled_mmfree 2>&1 | tee log10.log

# conda activate bitnet
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 0 --max_batch_power 0 --model_name microsoft/bitnet-b1.58-2B-4T 2>&1 | tee log5.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 0 --max_batch_power 0 --model_name microsoft/bitnet-b1.58-2B-4T --compression 2>&1 | tee log6.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 7 --max_batch_power 7 --model_name microsoft/bitnet-b1.58-2B-4T 2>&1 | tee log7.log
# python auto_profiler.py -s 1000 --max_new_tokens 2 --min_batch_power 7 --max_batch_power 7 --model_name microsoft/bitnet-b1.58-2B-4T --compression 2>&1 | tee log8.log

# cat /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_dataridger-MMfreeLM-2.7Bmin_batch1maxBatch1seqLen1000Compression:False.csv > /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2  /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_dataridger-MMfreeLM-2.7Bmin_batch1maxBatch1seqLen1000Compression:True.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2  /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_dataridger-MMfreeLM-2.7Bmin_batch256maxBatch256seqLen1000Compression:False.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2  /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_dataridger-MMfreeLM-2.7Bmin_batch256maxBatch256seqLen1000Compression:True.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv

# tail -n +2 /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_datamicrosoft-bitnet-b1.58-2B-4Tmin_batch1maxBatch1seqLen1000Compression:False.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2 /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_datamicrosoft-bitnet-b1.58-2B-4Tmin_batch1maxBatch1seqLen1000Compression:True.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2 /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_datamicrosoft-bitnet-b1.58-2B-4Tmin_batch128maxBatch128seqLen1000Compression:False.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
# tail -n +2 /home/eabban/eddy_matmulfreellm/outputs/csvs/roofline_datamicrosoft-bitnet-b1.58-2B-4Tmin_batch128maxBatch128seqLen1000Compression:True.csv >> /home/eabban/eddy_matmulfreellm/paper/paper_csv/roofline.csv
