"""
Evaluation with AutoAttack.

python eval-aa.py --fname_input xxx --eps_eval xxx --batch_size_for_eval xxx
"""

import json
import time
import tqdm
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import seed
from core.utils.utils import str2bool, str2float


# Setup

def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')
    
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--early_stopping', type=str2bool, default=True) 
    parser.add_argument('--gpu', type = int, default=2)

    return parser

parse = parser_eval()
args = parse.parse_args()

for experiment in tqdm.tqdm(os.listdir(args.folder)):
    print(experiment)
    if 'pert' in experiment:
        eps = int(experiment.split('pert')[1].split('_')[0])
    else:
        eps=8
    if os.path.exists(os.path.join(args.folder,experiment,'log-pgd50-10.log')):
        print('Already exists')
    elif os.path.exists(os.path.join(args.folder,experiment,'val_best.pt')):

        bash_cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python3 eval_pgd50-10.py --eps_eval {eps} --fname_input ' + os.path.join(args.folder,experiment) + f' --early_stopping {args.early_stopping}'
        os.system(bash_cmd)
