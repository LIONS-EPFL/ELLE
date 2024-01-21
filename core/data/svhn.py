import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms


DATA_DESC = {
    'data': 'svhn',
    'classes': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}


def load_svhn(data_dir, use_augmentation=False, validation = False):
    """
    Returns SVHN train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = test_transform
    
    train_dataset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
    test_dataset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=test_transform)
    if validation:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = 1024
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        return train_dataset, test_dataset, train_sampler, val_sampler
    return train_dataset, test_dataset