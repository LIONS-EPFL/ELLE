import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from .SLAT_hidden_module import HiddenPerturb

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, eps, alpha = 0.9):
        super(PreActResNet, self).__init__()
        print('Load PreAct')

        self.num_cls = num_classes
        self.eta = eps

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, self.num_cls)

        self.alpha = alpha

        self.noisy_module = nn.ModuleDict({
            'input': HiddenPerturb('advGNI', self.eta, self.alpha, True, normalize = False),
            'conv1': HiddenPerturb('advGNI', self.eta, 2.*self.alpha, normalize = False),# multiply alpha by 4 because we are not normalizing the images
            'layer1': HiddenPerturb('advGNI', self.eta, 2.*self.alpha, normalize = False),
            #'layer2': HiddenPerturb('advGNI', self.eta, self.alpha),
            #'layer3': HiddenPerturb('advGNI', self.eta, self.alpha),
            #'layer4': HiddenPerturb('advGNI', self.eta, self.alpha)
        })

        self.grads = {
            'input': None,
            'conv1': None,
            'layer1': None,
            #'layer2': None,
            #'layer3': None,
            #'layer4': None,
        }

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, add_adv=False, hook=False, init_hidden=False):
        x_hat = self.noisy_module['input'](x, self.grads['input'], add_adv, init_hidden)
        if hook:
            x_hat.register_hook(self.save_grad('input'))

        h = self.conv1(x_hat)
        if hook:
            h.register_hook(self.save_grad('conv1'))
        h = self.noisy_module['conv1'](h, self.grads['conv1'], add_adv, init_hidden)

        h = self.layer1(h)
        if hook:
            h.register_hook(self.save_grad('layer1'))
        h = self.noisy_module['layer1'](h, self.grads['layer1'], add_adv, init_hidden)

        h = self.layer2(h)
        """
        if hook:
            h.register_hook(self.save_grad('layer2'))
        h = self.noisy_module['layer2'](h, self.grads['layer2'], add_adv, init_hidden)
        """

        h = self.layer3(h)
        """
        if hook:
            h.register_hook(self.save_grad('layer3'))
        h = self.noisy_module['layer3'](h, self.grads['layer3'], add_adv, init_hidden)
        """

        h = self.layer4(h)
        """
        if hook:
            h.register_hook(self.save_grad('layer4'))
        h = self.noisy_module['layer4'](h, self.grads['layer4'], add_adv, init_hidden)
        """

        h = F.relu(self.bn(h))
        h = F.avg_pool2d(h, 4)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        return h

    def _yield_theta(self):
        b = []
        b.append(self.conv1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.bn)
        b.append(self.linear)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def optim_theta(self):
        return [{'params': self._yield_theta()}]


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


def PreActResNet18(num_classes = 10, eps = 8/255):
    model = PreActResNet(PreActBlock, [2,2,2,2], num_classes = num_classes, eps = eps)
    return model
