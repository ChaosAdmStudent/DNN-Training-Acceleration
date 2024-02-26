''' 
Loads CIFAR-10 dataset, transforms it and makes it ready for use by ML models. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 


classes = ('plane', 'car' , 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck') 


def get_data(batch_size:int, train:bool, download=True):  
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
    
    print('Creating Data Loader')
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)  
    print('Data Loader created!') 
    print('---------------')

    return loader 



