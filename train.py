import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 
import numpy as np 
import torchvision.models as models  
from preprocess import get_dataloader  
import time  
import argparse  
import os 

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

def setup(rank, world_size): 
    '''
    Setting up process group so that each process can communicate with each other for DDP. 
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 

def cleanup():
    dist.destroy_process_group()   

def ddp_train(rank, world_size, model, model_name, batch_size, num_epochs, lr, momentum, print_every_n):  
    setup(rank, world_size)  
    device = torch.device(f'cuda:{rank}')
    print(f'Running DDP on machine {rank} device {device}') 

    # Get the dataloader  
    train_loader = get_dataloader(batch_size, train=True, is_dist=True, rank=rank, world_size=world_size)

    # Move model to the GPU managed by the process 
    model.to(rank)   

    # Wrap model in DDP 
    model = DDP(model, device_ids=[rank])

    # Train the model     
    loss = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)  

    total_batches = len(train_loader)
    batch_time = AverageMeter()
    total_time = AverageMeter()
    n_correct = 0  
    n_samples = 0 
    
    print(f'---------------\nStarting training node {rank}!') 
    
    epoch_start = time.time() 
    for i in range(1,num_epochs+1):   
        train_loader.sampler.set_epoch(i)
        
        batch_start = time.time() 
        print_epoch = False 
        for j, (features, targets) in enumerate(train_loader):  
            features = features.to(device) 
            targets = targets.to(device) 
            y_pred = model(features) 

            if model_name == 'inception-v3': 
                y_pred, _ = y_pred

            l = loss(y_pred,targets)  
            _, predicted = torch.max(y_pred, 1) 
            n_correct += (predicted == targets).sum().item()  
            n_samples += targets.size(0)

            l.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

            batch_time.update(time.time() - batch_start) 
            batch_start = time.time() 

            # if j % (total_batches//print_every_n) == 0: 
            #     if not print_epoch: 
            #         print(f'Epoch {i}')
            #         print_epoch = True 

            #     print(f'\t step {j}/{total_batches} loss: {l.item():.4f} acc: {100*(n_correct/n_samples):.2f}%')  

        total_time.update(time.time() - epoch_start) 
        epoch_start = time.time() 

    
    print(f'Training completed by node: {rank}!') 
    print(f'Average batch processing time node {rank}: {batch_time.avg:.3f} seconds') 
    print(f'Average dataset processing time node {rank}: {total_time.avg:.3f} seconds') 
    print('---------------')
    cleanup()

    return model 
     
def train(model, model_name:str, batch_size ,print_every_n,num_epochs=5, lr = 0.001, momentum=0.9, model_save=False): 
    
    assert torch.cuda.is_available(), "DNN Acceleration not possible on CPU"
    device = torch.device('cuda') 

    train_loader = get_dataloader(batch_size=batch_size, train=True) 

    loss = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)  

    total_batches = len(train_loader)
    batch_time = AverageMeter()
    total_time = AverageMeter()
    n_correct = 0  
    n_samples = 0 
    
    print('Starting training!') 
    
    epoch_start = time.time() 
    for i in range(1,num_epochs+1): 
        batch_start = time.time() 
        print_epoch = False 
        for j, (features, targets) in enumerate(train_loader):  
            features = features.to(device) 
            targets = targets.to(device)
            y_pred = model(features) 

            if model_name == 'inception-v3': 
                y_pred, _ = y_pred

            l = loss(y_pred,targets)  
            _, predicted = torch.max(y_pred, 1) 
            n_correct += (predicted == targets).sum().item()  
            n_samples += targets.size(0)

            l.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

            batch_time.update(time.time() - batch_start) 
            batch_start = time.time() 

            if j % (total_batches//print_every_n) == 0: 
                if not print_epoch: 
                    print(f'Epoch {i}')
                    print_epoch = True 

                print(f'\t step {j}/{total_batches} loss: {l.item():.4f} acc: {100*(n_correct/n_samples):.2f}%')  

        total_time.update(time.time() - epoch_start) 
        epoch_start = time.time() 

    print('Training complete!') 
    print(f'Average batch processing time: {batch_time.avg:.3f} seconds') 
    print(f'Average dataset processing time: {total_time.avg:.3f} seconds') 
    
    if model_save: 
        torch.save(model.state_dict(), f'../model_{model_name}.pth')
    
    return model 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="Train a model.") 
    parser.add_argument('--mode', type=str, default='none',help='options: dp, ddp, mp, none') 
    args = parser.parse_args() 

    # Training Parameters
    batch_size = 20  
    num_epochs = 1 
    lr = 0.001 
    momentum = 0.9 
    print_every_n = 5 

    model_name = 'inception-v3' 
    model = get_pretrained_model(model_name)
    
    if torch.cuda.device_count() > 1:  
        print(f'Using {torch.cuda.device_count()} GPUs') 
        
        # Using DataParallel  
        if args.mode == 'dp':  
            print('Using DataParallel')
            model = nn.DataParallel(model)  
            model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=10) 

        # Using DistributedDataParallel 
        if args.mode == 'ddp':
            print('Using DistributedDataParallel') 
            world_size = torch.cuda.device_count()  
            mp.spawn(
                ddp_train, 
                args = (world_size, model, model_name, batch_size, num_epochs, lr, momentum, print_every_n), 
                nprocs = world_size, 
                join=True
            )   
    
