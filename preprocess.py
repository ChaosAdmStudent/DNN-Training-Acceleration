''' 
Loads CIFAR-10 dataset, transforms it and makes it ready for use by ML models. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms  
from torch.utils.data.distributed import DistributedSampler  
from torch.utils.data import DataLoader


classes = ('plane', 'car' , 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck') 


def get_dataset(batch_size:int, train:bool, download=True):  
    device = torch.device('cuda')
    transform = transforms.Compose([
        transforms.Resize(size=(299, 299)), 
        transforms.ToTensor(), 
        transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root= '../data', train = train,
        download =download, transform = transform
    )

    return dataset 

def get_dataloader(dataset, batch_size, is_dist:bool, rank=None, world_size = None):
    
    print('Creating Data Loader') 
    loader = None 

    if is_dist: 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank,drop_last=True)) 
    else: 
        loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)  
    print('Data Loader created!') 
    print('---------------')

    return loader 



