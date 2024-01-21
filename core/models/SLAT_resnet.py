import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SLAT_hidden_module import HiddenPerturb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config, block, num_blocks):
        super(ResNet, self).__init__()

        self.config = config
        self.data_name = config['dataset']['name']
        num_classes = config['dataset'][self.data_name]['num_cls']
        self.architecture = config['model']['baseline']
        self.eta = self.config['model']['ResNet']['eta']

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if 'advGNI' in self.architecture:
            self.alpha = config['model'][self.architecture]['alpha']
        else:
            self.alpha = 0.

        self.noisy_module = nn.ModuleDict({
            'input': HiddenPerturb(self.architecture, self.eta/255., self.alpha, True),
            'conv1': HiddenPerturb(self.architecture, self.eta/255., 2.*self.alpha),
            'block1': HiddenPerturb(self.architecture, self.eta/255, 2.*self.alpha),
            'block2': HiddenPerturb(self.architecture, self.eta/255., 2.*self.alpha),
            #'block3': HiddenPerturb(self.architecture, self.eta/255., self.alpha),
            #'block4': HiddenPerturb(self.architecture, self.eta/255., self.alpha),
        })

        self.grads = {
            'input': None,
            'conv1': None,
            'block1': None,
            'block2': None,
            #'block3': None,
            #'block4': None,
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

        h = F.relu(self.bn1(self.conv1(x)))
        if hook:
            h.register_hook(self.save_grad('conv1'))
        h = self.noisy_module['conv1'](h, self.grads['conv1'], add_adv, init_hidden)

        h = self.layer1(h)
        if hook:
            h.register_hook(self.save_grad('block1'))
        h = self.noisy_module['block1'](h, self.grads['block1'], add_adv, init_hidden)

        h = self.layer2(h)
        if hook:
            h.register_hook(self.save_grad('block2'))
        h = self.noisy_module['block2'](h, self.grads['block2'], add_adv, init_hidden)

        h = self.layer3(h)
        """
        if hook:
            h.register_hook(self.save_grad('block3'))
        h = self.noisy_module['block3'](h, self.grads['block3'], add_adv, init_hidden)
        """

        h = self.layer4(h)
        """
        if hook:
            h.register_hook(self.save_grad('block4'))
        h = self.noisy_module['block4'](h, self.grads['block4'], add_adv, init_hidden)
        """

        out = F.avg_pool2d(h, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet_depth(config, depth):
    if depth == 18:
        return ResNet(config, BasicBlock, [2, 2, 2, 2])
    elif depth == 34:
        return ResNet(config, BasicBlock, [3, 4, 6, 3])
    elif depth == 50:
        return ResNet(config, Bottleneck, [3, 4, 6, 3])
    elif depth == 101:
        return ResNet(config, Bottleneck, [3, 4, 23, 3])
    else:
        raise NotImplementedError