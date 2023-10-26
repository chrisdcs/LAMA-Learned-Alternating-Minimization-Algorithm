# LAMA by Chi Ding, University of Florida, Mathematics Department

"""
Train the LAMA model on the provided data.

Usage - Single-GPU training:
    $ python train.py --data_dir <path_to_data_dir> --model_dir <path_to_model_dir>

"""

import os
import argparse
import sys
from pathlib import Path


import torch
import torch.nn as nn


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # LAMA root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--ctcfg', type=str, default='', help='CT.yaml path')
    parser.add_argument('--initialization', type=str, default='', help='initialization type')