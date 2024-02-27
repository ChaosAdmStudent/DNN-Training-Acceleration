import torch
import torch.nn as nn
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP 
from preprocess import get_dataloader
import time   
import os 

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
    device = torch.device(f'cuda:{rank}')
    print(f'Running DDP on machine {rank} device {device}') 

    # Get the dataloader  
    train_loader = get_dataloader(batch_size, train=True, is_dist=True, rank=rank, world_size=world_size)

    # Move model to the GPU managed by the process 
    model.to(device)   

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
     
def train(model, model_name:str, batch_size,print_every_n, model_parallel:bool = False, num_epochs=5, lr = 0.001, momentum=0.9, model_save=False): 
    
    assert torch.cuda.is_available(), "DNN Acceleration not possible on CPU"
    device = torch.device('cuda') 

    train_loader = get_dataloader(batch_size=batch_size, train=True, is_dist=False) 

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
            if not model_parallel:
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
    
