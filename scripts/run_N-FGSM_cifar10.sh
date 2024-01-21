#!/bin/bash -l
cd ../
python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert26 --seed 1 \
--attack-iter 1 --attack-step 26/255 --attack-eps 26/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert24 --seed 1 \
--attack-iter 1 --attack-step 24/255 --attack-eps 24/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert22 --seed 1 \
--attack-iter 1 --attack-step 22/255 --attack-eps 22/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert20 --seed 1 \
--attack-iter 1 --attack-step 20/255 --attack-eps 20/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert18 --seed 1 \
--attack-iter 1 --attack-step 18/255 --attack-eps 18/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert16 --seed 1 \
--attack-iter 1 --attack-step 16/255 --attack-eps 16/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert14 --seed 1 \
--attack-iter 1 --attack-step 14/255 --attack-eps 14/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert12 --seed 1 \
--attack-iter 1 --attack-step 12/255 --attack-eps 12/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert10 --seed 1 \
--attack-iter 1 --attack-step 10/255 --attack-eps 10/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert8 --seed 1 \
--attack-iter 1 --attack-step 8/255 --attack-eps 8/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert6 --seed 1 \
--attack-iter 1 --attack-step 6/255 --attack-eps 6/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert4 --seed 1 \
--attack-iter 1 --attack-step 4/255 --attack-eps 4/255

python3 train.py --data 'cifar10'  --batch-size-validation 1024 --batch-size 128 \
--epochs 30 \
--lr 0.2  --scheduler 'GradAlign' --reg 'None' \
--output 'results_N-FGSM' --input_noise_rate 2.0 \
--model 'preact-resnet18' \
--fname_extra N-FGSM_seed1_pert2 --seed 1 \
--attack-iter 1 --attack-step 2/255 --attack-eps 2/255