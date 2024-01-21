# This module is adapted from https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/master/main_free.py
# Which in turn was adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
import pandas as pd
from utils import *
from validation import validate, validate_pgd
import torchvision.models as models
import torch.nn.functional as F

from apex import amp
import copy

def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def get_input_grad(model, X, y, opt, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    with amp.scale_loss(loss, opt) as scaled_loss: 
        scaled_loss.backward(create_graph=True if backprop else False,retain_graph=True if backprop else False)
    #loss.backward(create_graph=True if backprop else False)
    grad = delta.grad
    
    #grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--output_prefix', default='fast_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--elle_lambda', default=5000, type = float)
    parser.add_argument('--decay_rate', default=0.99, type = float)
    parser.add_argument('--sensitivity', default=2.0, type = float)
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name, configs.evaluate)
print_ = logger.info
cudnn.benchmark = True

def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value
    
    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))
    
    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    
    # Create the model
    if configs.pretrained:
        print_("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True)
    else:
        print_("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch]()
    # Wrap the model into DataParallel
    model.cuda()

    # reverse mapping
    param_to_moduleName = {}
    for m in model.modules():
        for p in m.parameters(recurse=False):
            param_to_moduleName[p] = str(type(m).__name__)

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    group_decay = [p for p in model.parameters() if 'BatchNorm' not in param_to_moduleName[p]]
    group_no_decay = [p for p in model.parameters() if 'BatchNorm' in param_to_moduleName[p]]
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0)]
    optimizer = torch.optim.SGD(groups, configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    if configs.TRAIN.half and not configs.evaluate:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print_("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print_("=> no checkpoint found at '{}'".format(configs.resume))
    
    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')
    
    resize_transform = []

    if configs.DATA.img_size > 0: 
        resize_transform = [ transforms.Resize(configs.DATA.img_size) ] 

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(resize_transform + [
            transforms.RandomResizedCrop(configs.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None)
    
    normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                    std=configs.TRAIN.std)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose( resize_transform + [
            transforms.CenterCrop(configs.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False,
        num_workers=configs.DATA.workers, pin_memory=True)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(val_loader, model, criterion, configs, logger)
        return
    
    lr_schedule = lambda t: np.interp([t], configs.TRAIN.lr_epochs, configs.TRAIN.lr_values)[0]
    
    if configs.TRAIN.adaptive:
        # if configs.resume:
        #     df = pd.read_csv(os.path.join(os.path.dirname(configs.resume), 'lambdas.csv'))
        #     configs.elle_values = list(df['lin_err'])
        #     configs.lambda_aux = configs.elle_lambda
        #     configs.elle_lambda = list(df['lambdas'])[-1]
        #     configs.lambdas = df.to_dict(orient='list')
        #     print(configs.elle_values)
        #     print(configs.elle_lambda)
        # else:
        configs.elle_values = []
        configs.lambda_aux = configs.elle_lambda
        if configs.ADV.clip_eps == 16/255:
            configs.elle_lambda = configs.elle_lambda
        else:
            configs.elle_lambda = 0
        print(f'initializing with {configs.elle_lambda}')
        configs.lambdas = {'epoch':[],'step':[],'lambdas':[], 'mu':[], 'sigma':[], 'lin_err':[]}

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, lr_schedule, configs.TRAIN.half, first = (epoch in [0,6,12] and configs.TRAIN.nfgsm))

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict()}, is_best, os.path.join('trained_models', configs.output_name),epoch + 1)
        if configs.TRAIN.adaptive:
            folder = os.path.join('trained_models', configs.output_name)
            pd_df = pd.DataFrame(configs.lambdas)
            pd_df.to_csv(os.path.join(folder,'lambdas.csv'),index = False)
    # Automatically perform PGD Attacks at the end of training
    # logger.info(pad_str(' Performing PGD Attacks '))
    # for pgd_param in configs.ADV.pgd_attack:
    #     validate_pgd(val_loader, val_model, criterion, pgd_param[0], pgd_param[1], configs, logger)

        
# Fast Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch, lr_schedule, half=False, first = False): 
    global global_noise_data
    crit = nn.CrossEntropyLoss(reduction='none')
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    
    if first:
        n_increase = 2500
        increase = configs.lambda_aux/n_increase
        configs.lambda_aux = 0
        configs.elle_lambda = 0

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        if configs.TRAIN.random_init: 
            global_noise_data.uniform_(-configs.ADV.clip_eps, configs.ADV.clip_eps)
        elif configs.TRAIN.nfgsm:
            #print('hey')
            global_noise_data.uniform_(-2*configs.ADV.clip_eps, 2*configs.ADV.clip_eps)

        for j in range(configs.ADV.n_repeats):
            # update learning rate
            lr = lr_schedule(epoch + (i*configs.ADV.n_repeats + j + 1)/len(train_loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True)#.cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            if half: 
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
            else:
                loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            if not configs.TRAIN.nfgsm:
                global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            # Descend on global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=False)#.cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            bs = input.shape[0]
            #ELLE loss
            if configs.TRAIN.elle:
                x_2 = input.repeat([2,1,1,1])
                # if not configs.TRAIN.nfgsm and configs.ADV.clip_eps == 16/255:
                #     x_2 = x_2 + 2*configs.ADV.clip_eps*(2*torch.rand(x_2.shape, device = input.device) - 1)               
                # else: 
                x_2 = x_2 + configs.ADV.clip_eps*(2*torch.rand(x_2.shape, device = input.device) - 1)
                x_2 = torch.clamp(x_2, 0, 1)
                x_2.sub_(mean).div_(std)
                alpha = torch.rand([bs,1,1,1],device = input.device)
                x_middle = (1-alpha)*x_2[:bs] + alpha*x_2[bs:]
                alpha = alpha.squeeze()
                x_adv = torch.cat((x_2,x_middle),dim=0)

                output = model(torch.cat([in1,x_adv],dim=0))
                #print(crit(output[2*bs:3*bs],target).shape, alpha.shape, (alpha*crit(output[2*bs:3*bs],target)).shape)
                lin_err = torch.nn.functional.smooth_l1_loss(crit(output[3*bs:],target), (1-alpha)*crit(output[bs:2*bs],target) + alpha*crit(output[2*bs:3*bs],target))
                if configs.TRAIN.adaptive:
                    configs.lambdas['mu'].append(np.mean(configs.elle_values))
                    configs.lambdas['sigma'].append(np.std(configs.elle_values))
                    if (len(configs.elle_values) > 2) and (lin_err > np.mean(configs.elle_values[-500:]) + configs.sensitivity*np.std(configs.elle_values[-500:])):
                        configs.elle_lambda = configs.lambda_aux
                    elif not (first and i <= n_increase):
                        configs.elle_lambda*=configs.decay_rate
                    configs.lambdas['lin_err'].append(lin_err.cpu().item())
                    configs.elle_values.append(lin_err.cpu().item())
                    configs.lambdas['lambdas'].append(configs.elle_lambda)
                    configs.lambdas['epoch'].append(epoch)
                    configs.lambdas['step'].append(i)
                #print(configs.elle_lambda)
                loss = criterion(output[:bs], target) + configs.elle_lambda*lin_err
            elif configs.TRAIN.gradalign:
                grad2 = get_input_grad(model, input, target, optimizer, configs.ADV.clip_eps, delta_init='random_uniform', backprop=True)
                grad1 = get_input_grad(model, input, target, optimizer, configs.ADV.clip_eps, delta_init='none', backprop=True)
                lin_err = (1- F.cosine_similarity(grad1.view(bs,-1),grad2.view(bs,-1))).mean()
                loss = criterion(output[:bs], target) + configs.elle_lambda*lin_err
            else:
                output = model(in1)
                loss = criterion(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            if half: 
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
            else: 
                loss.backward()

            optimizer.step()

            prec1, prec5 = accuracy(output[:bs], target, topk=(1, 5))
            if not torch.isnan(loss):
                losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if first and i < n_increase:
                configs.lambda_aux += increase
                if configs.decay_rate == 1:
                    configs.elle_lambda += increase

            if i % configs.TRAIN.print_freq == 0:
                print_('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'LR {lr:.3f}'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1,
                       top5=top5,cls_loss=losses, lr=lr))
                sys.stdout.flush()

if __name__ == '__main__':
    main()
