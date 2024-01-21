import json
import time
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data

from core.utils import format_time, Logger, seed, Trainer

from parser import parser_train

if __name__ == '__main__':
    parse = parser_train()
    args = parse.parse_args()

    fname_extra = args.fname_extra + '_{}Epoch'.format(args.epochs)

        
    if not args.resume:
        if args.data == 'cifar10':
            fname_extra = 'CIFAR10_' + fname_extra
        else:
            fname_extra = 'SVHN_' + fname_extra
        args.fname = os.path.join(args.output,fname_extra)

    if os.path.exists(args.fname) and not args.resume:
        print('\n\n\n\nThe file name already exists. Maybe check your hyperparameters or delete the file? {}\n\n\n'.format(args.fname))

    if not os.path.exists(args.fname) and not args.resume:
        os.makedirs(args.fname)

    if args.resume:
        # load all training parameter from file; later, use args.fname (args.resume_fname is meaningless after loading)
        args = torch.load(os.path.join(args.resume_fname, 'ResumeParameter.pth'))
        args.resume = True
        args.resume_fname = args.fname
        print(args)
    else:
        with open(os.path.join(args.fname, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
            
        torch.save(args, os.path.join(args.fname, 'ResumeParameter.pth') )

    DATA_DIR = os.path.join(args.data_dir, args.data)
    logger = Logger(os.path.join(args.fname, 'log-train.log'))
    # -


    info = get_data_info(DATA_DIR)
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_VALIDATION = args.batch_size_validation
    EPOCHS = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if args.debug:
        EPOCHS = 1
    # To speed up training
    torch.backends.cudnn.benchmark = True

    seed(args.seed)
    train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader = load_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, validation=True
    )
    logger.log('\n\n{}'.format(args))
    dataset = os.path.basename(os.path.normpath(DATA_DIR))
    del train_dataset, test_dataset, val_dataset

    seed(args.seed)
    trainer = Trainer(info, args)

    if EPOCHS > 0:
        metrics = pd.DataFrame()
        trainer.init_optimizer(args.epochs, len(train_dataloader))
            
        # best adv acc
        val_best = 0 
        test_best = 0
        
        start_epoch = 0


    # +
    if args.resume:
        checkpoint = torch.load(os.path.join(trainer.params.fname, 'latest_checkpoint.pt'))

        trainer.model.module.load_state_dict(checkpoint['unaveraged_model'])
        
        trainer.optimizer.load_state_dict(checkpoint['optimizer']) 
        trainer.scheduler.load_state_dict(checkpoint['scheduler']) 
        
        val_best = checkpoint['val_best']
        test_best = checkpoint['test_best']
        
        start_epoch = trainer.scheduler.last_epoch
        
        if trainer.params.scheduler not in ['cyclic']:
            assert checkpoint['epoch'] + 1 == start_epoch, 'Resuming: start_epoch is wrong! checkpoint_epoch + 1:{} start_epoch:{}'.format(checkpoint['epoch'] + 1, start_epoch)
        else:
            start_epoch = checkpoint['epoch'] + 1
        
        logger.log('Resuming from epoch {} with current val_best:{:.4f} and test_best:{:.4f} from {}'.format(\
                    start_epoch,val_best,test_best,args.fname))

    val_clean = trainer.eval(val_dataloader)*100
    val_adv = trainer.eval(val_dataloader, adversarial=True)*100
    logger.log('Val Clean: {:.4f}%\tRobust: {:.4f}%'.format(val_clean, val_adv))

    if args.data == 'svhn':
        #Gradualy increase epsilon in the first 5 epochs
        epsilon_max = args.attack_eps
        step_max = args.attack_step
        args.attack_eps = 0
        args.attack_step = 0
        
    if args.lambda_schedule == 'constant':
        lambdas = [args.lin_reg for i in range(EPOCHS)]
    elif args.lambda_schedule == 'onoff':
        trainer.elle_values = []
        trainer.args.lambda_aux = args.lin_reg
        trainer.args.lin_reg = 0
        
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        last_lr = trainer.scheduler.get_last_lr()[0]
        print('last_lr is :{}'.format(last_lr))
        print(trainer.params.fname)

        train_stat = trainer.train(train_dataloader, epoch=epoch, adversarial=True) # 'loss', 'clean_acc' and 'adversarial_acc'
        train_loss, train_clean_acc, train_adv_acc = train_stat['loss'], train_stat['clean_acc'], train_stat['adversarial_acc']
        finish = time.time()
        logger.log('Training Loss: {:.4f}.\tLR: {:.4f}'.format(train_loss, last_lr))

        test_clean_acc = trainer.eval(test_dataloader)
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)

        val_clean_acc = trainer.eval(val_dataloader)
        val_adv_acc = trainer.eval(val_dataloader, adversarial=True)

        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tVal: {:.2f}%.\tTest: {:.2f}%.'.format(\
                        train_clean_acc*100, val_clean_acc*100, test_clean_acc*100))
        logger.log('Robust Accuracy-\tTrain: {:.2f}%.\tVal: {:.2f}%.\tTest: {:.2f}%.'.format(\
                        train_adv_acc*100, val_adv_acc*100, test_adv_acc*100))
        if 'lin_err' in train_stat:
            epoch_metrics = {'train_loss':train_loss, 'train_lin_err': train_stat['lin_err'], 'train_clean_acc':train_clean_acc, 'train_adv_acc':train_adv_acc,\
                            'test_clean_acc':test_clean_acc, 'test_adv_acc':test_adv_acc,\
                            'val_clean_acc':val_clean_acc, 'val_adv_acc':val_adv_acc,\
                            'epoch': epoch, 'lr': last_lr} 
        else:
            epoch_metrics = {'train_loss':train_loss, 'train_clean_acc':train_clean_acc, 'train_adv_acc':train_adv_acc,\
                            'test_clean_acc':test_clean_acc, 'test_adv_acc':test_adv_acc,\
                            'val_clean_acc':val_clean_acc, 'val_adv_acc':val_adv_acc,\
                            'epoch': epoch, 'lr': last_lr}
        if val_adv_acc > val_best:
            val_best = val_adv_acc
            print('saving val_best: {:.2f}%'.format(val_best * 100))
            torch.save({'model': trainer.model.module.state_dict() if not args.SLAT else trainer.model.state_dict(),
                'val_best':val_best, 'epoch':epoch}, \
                os.path.join(trainer.params.fname, 'val_best.pt'))

        if test_adv_acc > test_best:
            test_best = test_adv_acc
            print('saving test_best: {:.2f}%'.format(test_best * 100))
        epoch_metrics.update({'train_time': finish-start})
            
        # save latest checkpoint
    
        torch.save({'model': trainer.model.module.state_dict() if not args.SLAT else trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
                'val_best':val_best, 'test_best':test_best, 'epoch':epoch}, \
                os.path.join(trainer.params.fname, 'latest_checkpoint.pt')) 
        
        if args.save_intermediate_models and epoch % args.save_intermediate_models == 0:
            if epoch > 0:
                torch.save({'model':trainer.model.module.state_dict() if not args.SLAT else trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
                    'val_best':val_best, 'test_best':test_best, 'epoch':epoch},\
                        os.path.join(trainer.params.fname, '{}.pt'.format(epoch))) 
        
        logger.log('Time taken: {}'.format(format_time(time.time()-start)))
        
        if epoch % 10 == 9:
            logger.log('\nCurrent Val_best: {:.2f}%\tTest_best: {:.2f}%\n'.format(val_best * 100,test_best * 100))
        
        metrics = pd.DataFrame(epoch_metrics, index=[0]) # each epoch, only hold metrics for this epoch and write to csv
        if epoch == 0:
            metrics.to_csv(os.path.join(args.fname, 'stats.csv'), mode='a', index=False, header=True)
        else:
            metrics.to_csv(os.path.join(args.fname, 'stats.csv'), mode='a', index=False, header=False)
        
        if args.data == 'svhn' and epoch <5:
            args.attack_eps += epsilon_max/5
            args.attack_step += step_max/5
            


    logger.log('\nTraining completed. Val_best: {:.2f}%\tTest_best: {:.2f}%'.format(val_best * 100,test_best * 100))
