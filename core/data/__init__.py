import os
import torch

from .cifar10 import load_cifar10
from .svhn import load_svhn

DATASETS = ['cifar10', 'svhn']

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
    'svhn': load_svhn
}


def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    elif 'svhn' in data_dir:
        from .svhn import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_dir, batch_size=256, batch_size_test=256, num_workers=4, use_augmentation=False, shuffle_train=True, 
              validation=False):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (bool): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]
        
    if validation:
        train_dataset, test_dataset, train_sampler, val_sampler = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation, validation=True)
  
    else:
        train_dataset, test_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation)
        print('\nNot using a validation set \n')
       
    pin_memory = torch.cuda.is_available()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                  num_workers=num_workers, pin_memory=pin_memory)
    if validation:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=pin_memory, sampler = train_sampler)
        val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=pin_memory, sampler = val_sampler)
        return train_dataset, test_dataset, val_dataloader.dataset, train_dataloader, test_dataloader, val_dataloader

    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                   num_workers=num_workers, pin_memory=pin_memory)

        return train_dataset, test_dataset, train_dataloader, test_dataloader
