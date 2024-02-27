''' 
Loads CIFAR-10 dataset, transforms it and makes it ready for use by ML models. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms   
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler  
from torch.utils.data import DataLoader


classes = ('plane', 'car' , 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck') 


def get_dataset(train:bool):  
    transform = transforms.Compose([
        transforms.Resize(size=(299, 299)), 
        transforms.ToTensor(), 
        transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root= '../data', train = train,
        download=True, transform = transform
    )

    return dataset 

def get_dataloader(batch_size, train:bool, is_dist:bool, rank=None, world_size = None):
    
    print(f'Creating Data Loader for node {rank}') 
    loader = None 
    
    dataset = get_dataset(train=train) 

    if is_dist: 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank,drop_last=True)) 
    else: 
        loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)  

    print(f'Data Loader created for node {rank}!') 
    print('---------------')

    return loader 

def get_pretrained_model(model:str): 
    valid_models = ['vgg16', 'inception-v3']  

    # Make sure CUDA is there, otherwise this wont be working anyways 
    assert torch.cuda.is_available() 
    device = torch.device('cuda')

    if not model in valid_models:
        print(f'Valid model options: {valid_models}')
        return None 

    if model == 'vgg16': 
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1') 
        in_features = model.classifier[6].in_features 
        out_features = 10 

        for param in model.parameters():
            param.requires_grad = False 

        model.classifier[6] = nn.Linear(in_features=in_features, out_features=out_features, bias=True) 
        model.to(device)   

    elif model == 'inception-v3': 
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')   
        in_features = model.fc.in_features 
        out_features = 10 

        for param in model.parameters():
            param.requires_grad = False 

        model.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True) 
        model.to(device)
    
    return model 



