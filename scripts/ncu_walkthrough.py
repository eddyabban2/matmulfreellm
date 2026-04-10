import subprocess
import argparse
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import csv
import pandas as pd

from pathlib import Path
sys.path.append('..')
import mmfreelm
import logging
from utils import CustomThread

logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, filename='roofline.log', filemode='w')
logger.setLevel(logging.DEBUG)

ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
logger.debug(f"Extracted ncu path: {ncu_path}")
os.chdir('../')

def run_ncu_profile(bs, new_tokens, seq_len):
    report_name = os.getcwd() + "/ncu_runs/walkthroughBatchSize"+ str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    benchmark_command = [
        ncu_path, "--nvtx",
        # "--nvtx", "--nvtx-include", "workload/HGRNBitAttentionForward/HGRNBitMLP/",
        "--nvtx-exclude", "warmup/",
        "--nvtx-include", "HGRNBitAttentionForward/",
        "--nvtx-include", "HGRNBitMLP/",
        "--config-file", "off",
        "--export", report_name,
        "--force-overwrite",
        "--replay-mode", "application",
        "--app-replay-match", "name",
        # "--target-processes", "all",
        "--target-processes", "application-only",
        "--metrics", metrics_string,
        "python", "quiet_run.py",
        "-b", str(bs),
        "-s", str(seq_len),
        "-n", str(new_tokens),
        "-i", "1"
    ]
    logger.debug(f"running command {' '.join(benchmark_command)}")
    # subprocess.run(benchmark_command, check=True, stdout=subprocess.DEVNULL)
    # subprocess.run(benchmark_command, check=True)