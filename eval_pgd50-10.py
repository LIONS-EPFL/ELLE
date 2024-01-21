"""
Evaluation with AutoAttack.

python eval-aa.py --fname_input xxx --eps_eval xxx --batch_size_for_eval xxx
"""

import json
import time
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack

from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import seed
from core.utils.utils import str2bool, str2float

def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)

def attack_pgd(model, X, y, eps, alpha, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data = delta + alpha * torch.sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta


def rob_acc(batches, model, eps, pgd_alpha, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, attack_iters, n_restarts, rs=rs,
                               verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        if corner:
            pgd_delta = clamp(X + eps * torch.sign(pgd_delta), 0, 1, cuda) - X
        pgd_delta_proj = clamp(X + eps * torch.sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)
    return robust_acc, avg_loss, pgd_delta_np


# Setup

def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')
    
    parser.add_argument('--norm_attack', type=str, default='Linf', choices = ['Linf', 'L2'])
    parser.add_argument('--eps_eval', type=float, default=8, help='Random seed.') # 8 for Linf, 0.5 for L2
    parser.add_argument('--fname_input', type=str, default='...')
    parser.add_argument('--batch_size_for_eval', type=int, default=1024) 
    parser.add_argument('--early_stopping', type=str2bool, default=True) 
    
 
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

parse = parser_eval()
args = parse.parse_args()

if args.norm_attack == 'Linf':
    eps_eval = args.eps_eval/255. # will use the eps specified by the parser_eval
else:
    eps_eval = args.eps_eval

pgd_alpha, pgd_alpha_rr = eps_eval/4, eps_eval/4
attack_iters, n_restarts = (50, 10)

# accessing and appending the args for training the model
with open(args.fname_input+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old) # new args = args from parser_eval and training args

DATA_DIR = args.data_dir + args.data
if args.early_stopping: 
    WEIGHTS = args.fname_input + '/val_best.pt'
else: 
    WEIGHTS = args.fname_input + '/latest_checkpoint.pt'

log_path = args.fname_input + '/log-pgd50-10.log'
logger = Logger(log_path)
logger.log('\n\n')

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_for_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)

logger.log('evaluation data size:{}'.format(y_test.size(0)))
# Model
# +
model = create_model(args.model, args.normalize, info, device,GroupNorm=args.GroupNorm) ## dataParallel
checkpoint = torch.load(WEIGHTS)

try:
    model.load_state_dict(checkpoint['unaveraged_model'])
except:
    model.module.load_state_dict(checkpoint['unaveraged_model']) # when checkpt is not dataParallel
model.eval()
# -

# AA Evaluation

# +
seed(args.seed)
if args.norm_attack == 'Linf':
    assert args.attack in ['fgsm', 'linf-pgd', 'linf-df', 'linf-apgd']
elif args.norm_attack == 'L2':
    assert args.attack in ['fgm', 'l2-pgd', 'l2-df', 'l2-apgd']
else:
    raise ValueError('Invalid norm_attack for evaluation')

adversary = AutoAttack(model, norm=args.norm_attack, eps=eps_eval, log_path=log_path, version=args.version, seed=args.seed)
# -

logger.log('{} AA evaluation on:\n{}\n'.format(args.norm_attack, WEIGHTS))
try:
    logger.log('epoch {} with val_best {}'.format(checkpoint['epoch'],checkpoint['val_best']))
except:
    logger.log('epoch {} with test_best {}'.format(checkpoint['epoch'],checkpoint['test_best']))
del checkpoint

logger.log('eps:{:.4f} batch size:{}\n'.format(eps_eval,BATCH_SIZE_VALIDATION))

adv_acc,_,_ = rob_acc(test_dataloader, model, eps_eval, pgd_alpha, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=True, cuda=True)

logger.log(f'robust accuracy: {adv_acc*100:.2f}%')
print ('Script Completed.')
