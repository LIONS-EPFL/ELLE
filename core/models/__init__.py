import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet
from .SLAT_preresnet import PreActResNet18
from .SLAT_wide_resnet import WideResNet_depth
from .SLAT_resnet import ResNet_depth

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish',
          'wrn-34-10-swish']


def create_model(name, normalize, info, device, patch = 4, SLAT = False, eps = 8/255):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    image_size = 0
    if info['data'] == 'tiny-imagenet':
        image_size = -1 #TBD
    elif 'cifar' in info['data']:
        image_size = 32
    elif info['data'] == 'imagenet':
        image_size = 224

    if not SLAT:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    else:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = PreActResNet18(num_classes=info['num_classes'], eps = eps)
            print('Using SLAT preact-resnet18')
        elif 'resnet' in name and 'preact' not in name:
            backbone = ResNet_depth(name, num_classes=info['num_classes'], eps = eps)
        elif 'wrn' in name and 'swish' not in name:
            backbone = WideResNet_depth(name, num_classes=info['num_classes'], eps = eps)
    ### replace BN
    if not SLAT:
        if normalize:
            model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
        else:
            model = torch.nn.Sequential(backbone)
    
        model = torch.nn.DataParallel(model)
    else:
        model = backbone
    model = model.to(device)
    return model
