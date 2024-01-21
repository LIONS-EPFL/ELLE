import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from math import sqrt

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

mu_t = torch.tensor(mean).view(3,1,1)
std_t = torch.tensor(std).view(3,1,1)

upper_limit = ((1. - mu_t)/ std_t)
lower_limit = ((0. - mu_t)/ std_t)

class HiddenPerturb(nn.Module):
    def __init__(self, architecture, eta, alpha_coeff=0.5, input=False, normalize = False):
        super(HiddenPerturb, self).__init__()
        self.architecture = architecture
        self.input = input
        self.alpha_coeff = alpha_coeff
        self.normalize = normalize
        if normalize:
            self.eta = eta / std_t if input else torch.tensor(eta)
        else:
            self.eta = torch.tensor(eta)

    def forward(self, x, grad_mask=None, add_adv=False, init_hidden=False):
        if self.training:
            if self.architecture == 'GNI':
                x_hat = x + torch.randn_like(x) * sqrt(0.001)
                return x_hat
            elif self.architecture in ['advGNI', 'advGNI_GA']:
                if init_hidden:
                    self.adv_noise = torch.zeros_like(x)
                if add_adv:
                    assert grad_mask is not None
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    self.adv_noise.data = self.adv_noise + sgn_mask * self.eta * self.alpha_coeff
                    self.lower_limit = lower_limit if self.input and self.normalize else x.amin((2,3)).view(x.shape[0], x.shape[1], 1, 1)
                    self.upper_limit = upper_limit if self.input and self.normalize else x.amax((2,3)).view(x.shape[0], x.shape[1], 1, 1)

                    # Adversarial constraint and Domain constraint
                    self.adv_noise.data = clamp(self.adv_noise, -self.eta, self.eta)
                    self.adv_noise.data = clamp(self.adv_noise, self.lower_limit - x, self.upper_limit - x)

                    self.adv_noise = self.adv_noise.detach()
                    x_hat = x + self.adv_noise
                    return x_hat
                else:
                    return x

            elif self.architecture == 'FGSM' or self.architecture == 'FGSM_GA':
                if add_adv and self.input:
                    grad_mask = grad_mask.detach()

                    with torch.no_grad():
                        sgn_mask = grad_mask.data.sign()

                    adv_noise = sgn_mask * self.eta
                    adv_noise.data = clamp(adv_noise, lower_limit - x, upper_limit - x)
                    adv_noise = adv_noise.detach()

                    x_hat = x + adv_noise
                    return x_hat
                else:
                    return x
            elif self.architecture == 'FGSM_RS':
                """Wong et al., ICLR 2020"""
                if self.input:
                    if not add_adv:
                        # Initialize delta
                        self.delta = torch.zeros_like(x)
                        for j in range(len(self.eta)):
                            self.delta[:, j, :, :].uniform_(-self.eta[j][0][0].item(), self.eta[j][0][0].item())
                            self.delta.data = clamp(self.delta, lower_limit - x, upper_limit - x)
                        self.delta.requires_grad = True
                        x_hat = x + self.delta[:x.size(0)]
                        return x_hat
                    else:
                        grad = self.delta.grad.detach()
                        self.delta.data = clamp(self.delta + 1.25*self.eta*torch.sign(grad), -self.eta, self.eta)
                        self.delta.data[:x.size(0)] = clamp(self.delta[:x.size(0)], lower_limit - x, upper_limit - x)
                        self.delta = self.delta.detach()
                        x_hat = x + self.delta[:x.size(0)]
                        return x_hat
                else:
                    return x
            else:
                # PGD will be included in noise.py
                return x
        else:
            return x