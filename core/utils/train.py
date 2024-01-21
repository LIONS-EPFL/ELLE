import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model

from .context import ctx_noparamgrad_and_eval
from .utils import seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew', 'GradAlign']


'''
GradAlign helper functions
'''
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
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad


# +
class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(Trainer, self).__init__()
        
        seed(args.seed)
        self.model = create_model(args.model, args.normalize, info, device, SLAT = args.SLAT, eps = args.attack_eps)

        self.params = args
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                         10, self.params.attack_eps/8)

        num_samples = 50000 if 'cifar' in self.params.data else 73257
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
    
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs, lr_steps):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs,lr_steps)
    
        
    def init_scheduler(self, num_epochs, lr_steps):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)]) 
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[int(num_epochs*0.2),int(num_epochs*0.4),int(num_epochs*0.6), int(num_epochs*0.8)]) 
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        elif self.params.scheduler == 'GradAlign':
            lr_steps = self.params.epochs * lr_steps #196 is the number of steps in a cifar10 epoch
            #lr_steps = self.params.epochs
            if self.args.data == 'svhn':
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0, max_lr=self.params.lr,
                    step_size_up=lr_steps * 2 / 5, step_size_down=lr_steps * 3 / 5)
            else:
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0, max_lr=self.params.lr,
                    step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

        else:
            self.scheduler = None
            raise ValueError('Invalid scheduler name')
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()

        '''
        dataframe to store the correlations:
        keys: param name form torch.named_parameters()
        values: cosine similarity of the gradients
        '''
        if self.args.lambda_schedule == 'onoff':
            lambdas = {'epoch':[],'step':[],'lambdas':[], 'mu':[], 'sigma':[], 'lin_err':[]}
        if self.args.track_logits_alignment or self.args.track_loss_alignment or self.args.track_lin_err:
            grad_alignments = {'epoch': [], 'step': [], 'loss_align': [], 'logits_align': [], 'loss_lin_err':[], 'logits_lin_err':[], f'loss_lin_err_{self.args.n_triplets}':[], 'loss_lin_err_0.5':[], 'loss_lin_err_max_curve':[], 'min_alpha':[],'max_alpha':[],'mean_alpha':[]}
        times = {'epoch': [], 'step': [], 'one_step': [], 'backward':[]}
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            update_iter += 1

            x, y = data
            x, y = x.to(device), y.to(device)
            '''
            gradient alignment metrics
            '''
            if self.args.track_logits_alignment or self.args.track_loss_alignment or self.args.track_lin_err:
                grad_alignments['epoch'].append(epoch)
                grad_alignments['step'].append(update_iter)
                if self.args.track_loss_alignment:
                    loss_alignment = self.loss_grad_alignment(self.model, x, y, self.args.attack_eps)
                    grad_alignments['loss_align'].append(loss_alignment)
                else:
                    grad_alignments['loss_align'].append(None)
                if self.args.track_logits_alignment:
                    logits_alignment = self.logits_grad_alignment(self.model, x, y, self.args.attack_eps)
                    grad_alignments['logits_align'].append(logits_alignment)
                else:
                    grad_alignments['logits_align'].append(None)
                if self.args.track_lin_err:
                    logits_lin_err, loss_lin_err = self.lin_err(self.model, x, y, self.args.attack_eps)
                    if self.args.n_triplets == 1:
                         grad_alignments['loss_lin_err_1'].append(loss_lin_err)
                    grad_alignments['loss_lin_err'].append(loss_lin_err)
                    grad_alignments['logits_lin_err'].append(logits_lin_err)
                else:
                    grad_alignments['loss_lin_err'].append(None)
                    grad_alignments['logits_lin_err'].append(None)
                if self.args.n_triplets>1:
                    _, loss_lin_err = self.lin_err(self.model, x, y, self.args.attack_eps, n_triplets=self.args.n_triplets)
                    grad_alignments[f'loss_lin_err_{self.args.n_triplets}'].append(loss_lin_err)
                if self.args.track_lin_err_05:
                    _, loss_lin_err = self.lin_err(self.model, x, y, self.args.attack_eps, alpha_type = 'middle')
                    grad_alignments['loss_lin_err_0.5'].append(loss_lin_err)
                else:
                    grad_alignments['loss_lin_err_0.5'].append(None)
                if self.args.track_lin_err_max_curve:
                    loss_lin_err, min_alpha, max_alpha, mean_alpha = self.lin_err(self.model, x, y, self.args.attack_eps, alpha_type = 'max_curve')
                    grad_alignments['loss_lin_err_max_curve'].append(loss_lin_err)
                    grad_alignments['min_alpha'].append(min_alpha)
                    grad_alignments['max_alpha'].append(max_alpha)
                    grad_alignments['mean_alpha'].append(mean_alpha)
                else:
                    grad_alignments['loss_lin_err_max_curve'].append(None)
                    grad_alignments['min_alpha'].append(None)
                    grad_alignments['max_alpha'].append(None)
                    grad_alignments['mean_alpha'].append(None)
            start_step = time.time()
            if self.args.SLAT:
                loss, batch_metrics = self.SLAT_loss(x, y)
            elif self.args.BAT:
            	loss, batch_metrics = self.BAT_loss(x, y)
            elif self.args.GAT:
            	loss, batch_metrics = self.GAT_loss(x, y)
            else:
                loss, batch_metrics = self.loss(x,y)
            finish_step = time.time()
            start_backward = time.time()
            loss.backward()
            finish_backward = time.time()
            times['epoch'].append(epoch)
            times['step'].append(update_iter)
            times['one_step'].append(finish_step-start_step)
            times['backward'].append(finish_backward-start_backward)
            if (self.args.lambda_schedule == 'onoff') and ('lin_err' in batch_metrics.keys()):
                lambdas['epoch'].append(epoch)
                lambdas['step'].append(update_iter)
                lambdas['lambdas'].append(self.args.lin_reg)
                lambdas['mu'].append(np.mean(self.elle_values))
                lambdas['sigma'].append(np.std(self.elle_values))
                lambdas['lin_err'].append(batch_metrics['lin_err'])
                if (len(self.elle_values) > 2) and (batch_metrics['lin_err'] > np.mean(self.elle_values) + self.args.sensitivity*np.std(self.elle_values)):
                    self.args.lin_reg = self.args.lambda_aux
                else:
                    self.args.lin_reg*=self.args.decay_rate
                self.elle_values.append(batch_metrics['lin_err'])
            
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic', 'GradAlign']:
                self.scheduler.step()
            
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])])
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        if self.args.lambda_schedule == 'onoff':
            if os.path.isfile(f'{self.args.fname}/lambdas.csv'):
                df = pd.read_csv(f'{self.args.fname}/lambdas.csv')
                pd_df = pd.DataFrame(lambdas)
                pd.concat((df,pd_df)).to_csv(f'{self.args.fname}/lambdas.csv',index = False)
            else:
                pd_df = pd.DataFrame(lambdas)
                pd_df.to_csv(f'{self.args.fname}/lambdas.csv',index = False)
        
        if self.args.track_logits_alignment or self.args.track_loss_alignment or self.args.track_lin_err:
            if os.path.isfile(f'{self.args.fname}/grad_alignments.csv'):
                df = pd.read_csv(f'{self.args.fname}/grad_alignments.csv')
                pd_df = pd.DataFrame(grad_alignments)
                pd.concat((df,pd_df)).to_csv(f'{self.args.fname}/grad_alignments.csv',index = False)
            else:
                pd_df = pd.DataFrame(grad_alignments)
                pd_df.to_csv(f'{self.args.fname}/grad_alignments.csv',index = False)
        
        if os.path.isfile(f'{self.args.fname}/times.csv'):
            df = pd.read_csv(f'{self.args.fname}/times.csv')
            pd_df = pd.DataFrame(times)
            pd.concat((df,pd_df)).to_csv(f'{self.args.fname}/times.csv',index = False)
        else:
            pd_df = pd.DataFrame(times)
            pd_df.to_csv(f'{self.args.fname}/times.csv',index = False)

        return dict(metrics.mean())

    def get_triplets(self,x,y,n_triplets):
        bs = x.shape[0]
        x_2 = x.repeat([2,1,1,1]) 
        x_2 = x_2 + self.args.attack_eps*(2*torch.rand(x_2.shape, device = x.device) - 1)
        alpha = torch.rand([x.shape[0],1,1,1],device = x.device)
        x_middle = (1-alpha)*x_2[:bs] + alpha*x_2[bs:]
        alpha = alpha.squeeze()
        x_adv = torch.cat((x_2,x_middle),dim=0)
        for i in range(n_triplets-1):
            x_2 = x.repeat([2,1,1,1]) 
            x_2 = x_2 + self.args.attack_eps*(2*torch.rand(x_2.shape, device = x.device) - 1)
            alphai = torch.rand([x.shape[0],1,1,1],device = x.device)
            x_middle = (1-alphai)*x_2[:bs] + alphai*x_2[bs:]
            alpha = torch.cat((alpha,alphai.squeeze()),dim=0)
            x_adv = torch.cat((x_adv,x_2,x_middle),dim=0)
        return x_adv, alpha

    def loss(self, x, y):
        '''
        generalization of AT, every variant studied in the paper is capured here, e.g.,
        if input_noise_rate==2, clamp = False and attack_iter=1 we have N-FGSM
        '''
        bs = x.shape[0]
        self.optimizer.zero_grad()
        x_adv = copy.deepcopy(x) + self.args.input_noise_rate*self.args.attack_eps*(2*torch.rand(x.shape, device = x.device) - 1)
        for i in range(self.args.attack_iter):
            x_adv.requires_grad=True
            outputs = self.model(x_adv)
            self.criterion(outputs,y).backward(retain_graph=True)
            grads_input = copy.deepcopy(x_adv.grad)
            x_adv = x_adv + self.args.attack_step*torch.sign(grads_input)

            if self.args.clamp:
                x_adv = torch.min(torch.max(x_adv, x - self.args.attack_eps), x + self.args.attack_eps).detach()
            self.model.zero_grad()
        
        x_adv.detach()

        out2 = self.model(x_adv)
        loss = self.criterion(out2,y)

        if self.args.reg == 'GradAlign':
            grad1 = get_input_grad(self.model, x, y, self.optimizer, self.args.attack_eps, delta_init='none', backprop=False)
            grad2 = get_input_grad(self.model, x, y, self.optimizer, self.args.attack_eps, delta_init='random_uniform', backprop=True)
            lin_err = (1- F.cosine_similarity(grad1.view(bs,-1),grad2.view(bs,-1))).mean()
            loss += self.args.lin_reg*lin_err
        elif self.args.reg == 'ELLE':
            # x_lin, alpha = self.get_triplets(x,y,self.args.n_triplets)
            # out = self.model(torch.cat((x_adv,x_lin),dim=0))
            # lin_err = 0
            # for i in range(self.args.n_triplets):
            #     criterion = nn.CrossEntropyLoss(reduction='none')
            #     lin_err += self.mse(criterion(out[3*(i+1)*bs:3*(i+1)*bs + bs],y), (1-alpha[bs*i:bs*(i+1)])*criterion(out[3*i*bs + bs:3*i*bs + 2*bs],y) + alpha[bs*i:bs*(i+1)]*criterion(out[3*i*bs + 2*bs:3*(i+1)*bs],y))/self.args.n_triplets
            # Get random triplet
            bs = x.shape[0]
            x_ab = x.repeat([2,1,1,1]) 
            x_ab = x_ab + self.args.attack_eps*(2*torch.rand(x_ab.shape, device = x.device) - 1)
            alpha = torch.rand([bs,1,1,1],device = x.device)
            x_c = (1-alpha)*x_ab[:bs] + alpha*x_ab[bs:]
            alpha = alpha.squeeze()

            # Forward pass
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            losses = criterion(self.model(torch.cat((x_ab,x_c),dim=0)), y.repeat([3]))

            # Regularization term
            mse = torch.nn.MSELoss()
            lin_err = mse(losses[2*bs:], (1-alpha)*losses[:bs] + alpha*losses[bs:2*bs])
            loss += self.args.lin_reg*lin_err
        elif self.args.reg == 'LLR':
            lin_err = 0
            x_2 = x + self.args.attack_eps*(2*torch.rand(x.shape, device = x.device) - 1)
            x_3 = x + self.args.attack_eps*(2*torch.rand(x.shape, device = x.device) - 1)

            g = get_input_grad(self.model, x_2, y, self.optimizer, self.args.attack_eps, delta_init='none', backprop=True)
            out = self.model(torch.cat((x_2,x_3),dim=0))
            criterion = nn.CrossEntropyLoss(reduction='none')
            lin_err = self.mse(criterion(out[bs:], y), criterion(out[:bs], y) + ((x_3-x_2)*g).sum(dim=[1,2,3]))
            loss += self.args.lin_reg*lin_err

        elif self.args.reg == 'CURE':
            g_2 = get_input_grad(self.model, x, y, self.optimizer, self.args.attack_eps, delta_init='none', backprop=False)
            g_3 = get_input_grad(self.model, x_adv, y, self.optimizer, self.args.attack_eps, delta_init='none', backprop=True)

            lin_err = ((g_2-g_3)*(g_2-g_3)).mean(dim=0).sum()
            loss += self.args.lin_reg*lin_err

        preds = out2.detach()
        batch_metrics = {'loss': loss.item()}
        if self.args.reg != 'None':
             batch_metrics.update({'lin_err': lin_err.item()})
        preds_clean = self.model(x).detach()
        batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
        
    def SLAT_loss(self, x, y):
        x.requires_grad = True
        self.optimizer.zero_grad()
        #print(self.model.__dict__)
        out = self.model(x, hook=True,
                                init_hidden=True,
                                add_adv=False)
        loss = self.criterion(out, y)
        loss.backward()
        self.model.zero_grad()
        logit_adv = self.model(x, add_adv=True)

        # Main loss with adversarial example
        loss = self.criterion(logit_adv, y)
        preds = logit_adv.detach()
        batch_metrics = {'loss': loss.item()}
        preds_clean = out.detach()
        batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y,preds)})    
        return loss, batch_metrics
    
    def Guided_Attack(self,image,target,eps=8/255,bounds=[0,1],steps=1,P_out=[],l2_reg=10,alt=1): 
        tar = target.cuda()
        img = image.cuda()
        eps = eps/steps 
        for step in range(steps):
            img.requires_grad = True
            out  = self.model(img)
            R_out = nn.Softmax(dim=1)(out)
            cost = self.criterion(out,tar) + alt*l2_reg*(((P_out - R_out)**2.0).sum(1)).mean(0) 
            cost.backward()
            per = eps * torch.sign(img.grad.data)
            adv = img.data + per.cuda() 
            img = torch.clamp(adv,bounds[0],bounds[1])
        return img

    def GAT_loss(self, x, y):
        '''
        add regularization term enforcing local linearity
        '''
        bs = x.shape[0]

        self.optimizer.zero_grad()
        P_out = nn.Softmax(dim=1)(self.model(x))
        adv_data = x + (self.args.attack_eps*torch.sign(torch.tensor([0.5],device=x.device) - torch.rand_like(x))).cuda()
        adv_data = torch.clamp(adv_data,0.0,1.0).cuda()
        x_adv = self.Guided_Attack(adv_data,y,eps=self.args.attack_eps,bounds=[0,1],steps=1,P_out=P_out,l2_reg=10,alt=1)

        lin_err = 0
        x_lin, alpha = self.get_triplets(x,y,self.args.n_triplets)
        out = self.model(torch.cat((x_adv,x_lin),dim=0))
        lin_err = 0
        for i in range(self.args.n_triplets):
            criterion = nn.CrossEntropyLoss(reduction='none')
            lin_err += self.mse(criterion(out[3*(i+1)*bs:3*(i+1)*bs + bs],y), (1-alpha[bs*i:bs*(i+1)])*criterion(out[3*i*bs + bs:3*i*bs + 2*bs],y) + alpha[bs*i:bs*(i+1)]*criterion(out[3*i*bs + 2*bs:3*(i+1)*bs],y))/self.args.n_triplets
        
        P_out = nn.Softmax(dim=1)(self.model(x))
        Q_out = nn.Softmax(dim=1)(out[:bs])
        
        reg_loss =  ((P_out - Q_out)**2.0).sum(1).mean(0)

        loss = self.criterion(out[:bs], y) + 10.0*reg_loss + self.args.lin_reg*lin_err
        preds = out[:bs].detach()
        batch_metrics = {'loss': loss.item(), 'lin_err': lin_err.item()}      
        preds_clean = out[bs:2*bs].detach()
        batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics

    def loss_grad_alignment(self,model,x,y,eps):
        '''
        compute GradAlign regularization term
        '''
        bs = x.shape[0]
        grad1 = get_input_grad(self.model, x, y, self.optimizer, self.args.attack_eps, delta_init='none', backprop=False)
        grad2 = get_input_grad(self.model, x, y, self.optimizer, self.args.attack_eps, delta_init='random_uniform', backprop=False)
        #print(grad.shape, F.cosine_similarity(grad.view(bs,-1),grad2.view(bs,-1)).shape)
        alignment = F.cosine_similarity(grad1.view(bs,-1),grad2.view(bs,-1)).mean()
        model.zero_grad()
        
        x.requires_grad=False
        return alignment.item()

    def logits_grad_alignment(self,model,x,y,eps):
        '''
        compute gradient alignment as in GradAlign but with the output of a random position of the logits.
        '''
        bs = x.shape[0]
        model.zero_grad()
        x_2 = x 
        x.requires_grad = True
        x_2.requires_grad = True
        out = self.model(torch.cat((x,x_2),dim=0))
        randi = torch.randint(high = out.shape[1], size=[1]).item()

        grad,grad2 = torch.autograd.grad(out[:,randi].sum(), [x,x_2])
        alignment = F.cosine_similarity(grad.view(bs,-1),grad2.view(bs,-1)).mean()
        model.zero_grad()

        x.requires_grad=False
        x_2.requires_grad=False
        return alignment.item()

    def lin_err(self,model,x,y,eps, n_triplets = 1, alpha_type = 'rand'):
        '''
        compute linear error of both the loss and the logits
        '''
        bs = x.shape[0]
        if alpha_type == 'rand':
            x_lin, _, _, alpha = self.expanded_perturbed_dataset(x,y,nreps = 2,eps = self.args.attack_eps, noise = 'rand_3_points', n_triplets = n_triplets)
            with torch.no_grad():
                out = self.model(x_lin)
                logits_lin_err = 0
                loss_lin_err = 0
                for i in range(self.args.n_triplets):
                    logits_lin_err += self.mse(out[3*i*bs + 2*bs:3*(i+1)*bs], (1-alpha[bs*i:bs*(i+1)].unsqueeze(-1))*out[3*i*bs:3*i*bs + bs] + alpha[bs*i:bs*(i+1)].unsqueeze(-1)*out[3*i*bs + bs:3*(i)*bs + 2*bs])/self.args.n_triplets
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    loss_lin_err += self.mse(criterion(out[3*i*bs + 2*bs:3*(i+1)*bs],y), (1-alpha[bs*i:bs*(i+1)])*criterion(out[3*i*bs:3*i*bs + bs],y) + alpha[bs*i:bs*(i+1)]*criterion(out[3*i*bs + bs:3*(i)*bs + 2*bs],y))/self.args.n_triplets
                return logits_lin_err.cpu().item(), loss_lin_err.cpu().item()
        elif alpha_type == 'middle':
            x_2 = x.repeat([2,1,1,1]) 
            y_adv = y.repeat([3])
            x_2 = x_2 + eps*(2*torch.rand(x_2.shape, device = x.device) - 1)
            x_2 = torch.clamp(x_2, 0, 1)
            alpha = 0.5
            x_middle = (1-alpha)*x_2[:bs] + alpha*x_2[bs:]
            x_lin = torch.cat((x_2,x_middle),dim=0)
            with torch.no_grad():
                out = self.model(x_lin)
                logits_lin_err = self.mse(out[2*bs:], (1-alpha)*out[:bs] + alpha*out[bs:2*bs])
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_lin_err = self.mse(criterion(out[2*bs:],y), (1-alpha)*criterion(out[:bs],y) + alpha*criterion(out[bs:2*bs],y))
                return logits_lin_err.cpu().item(), loss_lin_err.cpu().item()
        elif alpha_type == 'max_curve':
            x_2 = x.repeat([2,1,1,1]) 
            y_adv = y.repeat([3])
            x_2 = x_2 + eps*(2*torch.rand(x_2.shape, device = x.device) - 1)
            x_2 = torch.clamp(x_2, 0, 1)
            alpha = 0.5*torch.ones([bs,1,1,1], device = x.device)
            lr = 0.075
            for i in range(10):
                alpha.requires_grad=True
                x_middle = (1-alpha)*x_2[:bs] + alpha*x_2[bs:]
                x_lin = torch.cat((x_2,x_middle),dim=0)
                out = self.model(x_lin)
                criterion = nn.CrossEntropyLoss(reduction='none')
                lin_err = self.mse(criterion(out[2*bs:],y), (1-alpha.squeeze())*criterion(out[:bs],y) + alpha.squeeze()*criterion(out[bs:2*bs],y))
                lin_err.backward(retain_graph=True)
                #print(lin_err.item())
                alpha = torch.clamp(alpha +lr*torch.sign(alpha.grad),0.01,0.99).detach()
                self.model.zero_grad()
            
            return lin_err.cpu().item(), alpha.min().cpu().item(), alpha.max().cpu().item(), alpha.mean().cpu().item()
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc

    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
