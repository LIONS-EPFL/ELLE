#!/bin/bash -l
cd ../
python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert26 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 26/255 --attack-eps 26/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert24 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 24/255 --attack-eps 24/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert22 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 22/255 --attack-eps 22/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert20 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 20/255 --attack-eps 20/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert18 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 18/255 --attack-eps 18/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert16 --lin_reg 2.0 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 16/255 --attack-eps 16/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert14 --lin_reg 1.124 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 14/255 --attack-eps 14/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert12 --lin_reg 0.632 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 12/255 --attack-eps 12/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert10 --lin_reg 0.356 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 10/255 --attack-eps 10/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert8 --lin_reg 0.200 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 8/255 --attack-eps 8/255  

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert6 --lin_reg 0.112 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 6/255 --attack-eps 6/255  

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert4 --lin_reg 0.063 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 4/255 --attack-eps 4/255  

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'GradAlign' \
--output 'results_GradAlign' \
--model 'preact-resnet18' \
--fname_extra GradAlign_seed1_pert2 --lin_reg 0.036 --seed 1 --lambda_schedule 'constant' \
--attack-iter 1 --attack-step 2/255 --attack-eps 2/255  