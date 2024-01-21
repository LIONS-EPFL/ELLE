#!/bin/bash -l
cd ../
python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'CURE' \
--output 'results_CURE' \
--model 'preact-resnet18' \
--fname_extra CURE_seed1_pert16 --lin_reg 50000 --seed 1 --lambda_schedule 'onoff' \
--attack-iter 1 --attack-step 16/255 --attack-eps 16/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'CURE' \
--output 'results_CURE' \
--model 'preact-resnet18' \
--fname_extra CURE_seed1_pert8 --lin_reg 1000 --seed 1 --lambda_schedule 'onoff' \
--attack-iter 1 --attack-step 8/255 --attack-eps 8/255  
