import argparse 
import time 
import matplotlib.pyplot as plt  
import numpy as np
import torch 
import torch.nn as nn 
import torch.multiprocessing as mp  
from preprocess import get_pretrained_model, get_dataloader 
from train import train, ddp_train  
from model_parallel import ModelParallelVGG16_2GPU, ModelParallelVGG16_3GPU, ModelParallelVGG16_4GPU

def get_comms_time_model_parallel(): 
    print('Analyzing Model Parallelism\n') 
    
    plt.figure(figsize=(10,6))
    data_loader_32 = get_dataloader(batch_size=32, train=True, is_dist=False) 
    data_loader_64 = get_dataloader(batch_size=64, train=True, is_dist=False) 

    labels = ['4 GPUs', '3 GPUs', '2 GPUs'] 
    x_pos = np.arange(len(labels))
    width = 0.3 
    overheads_32 = []  
    overheads_64 = [] 

    vgg16_4gpu = ModelParallelVGG16_4GPU(debugging=True) #MP with 4 GPUs 
    vgg16_3gpu = ModelParallelVGG16_3GPU(debugging=True) #MP with 3 GPUs
    vgg16_2gpu = ModelParallelVGG16_2GPU(debugging=True) #MP with 2 GPUs

    # Compute loading time for 32 batch size
    for features,targets in data_loader_32: 
        vgg16_4gpu(features) 
        vgg16_3gpu(features) 
        vgg16_2gpu(features) 
    
    overheads_32.extend([vgg16_4gpu.get_overhead(), vgg16_3gpu.get_overhead(), vgg16_2gpu.get_overhead()]) 
    
    vgg16_4gpu.reset_overhead()  
    vgg16_3gpu.reset_overhead()  
    vgg16_2gpu.reset_overhead()        

    # Compute loading time for 64 batch size
    for features,targets in data_loader_64: 
        vgg16_4gpu(features) 
        vgg16_3gpu(features) 
        vgg16_2gpu(features) 

    overheads_64.extend([vgg16_4gpu.get_overhead(), vgg16_3gpu.get_overhead(), vgg16_2gpu.get_overhead()]) 

    print('----------------') 
    print('Comms time for batch size 32: ') 
    print(f'\t 4 GPUs: {overheads_32[0]}')
    print(f'\t 3 GPUs: {overheads_32[1]}')
    print(f'\t 2 GPUs: {overheads_32[2]}')  

    print('Comms time for batch size 64: ') 
    print(f'\t 4 GPUs: {overheads_64[0]}')
    print(f'\t 3 GPUs: {overheads_64[1]}')
    print(f'\t 2 GPUs: {overheads_64[2]}') 
    print('----------------') 

    # Plotting 
    plt.bar(x_pos - width/2, np.array(overheads_32), width, label='Batch_size = 32', color='black') 
    plt.bar(x_pos + width/2, np.array(overheads_64), width, label='Batch_size = 64', color='orange')
    plt.xticks(x_pos, labels)  
    plt.ylabel('Time (ms)') 
    plt.title('Model Parallelism Communication Time')
    plt.legend() 
    plt.show() 
    plt.savefig('../model_parallel.png', bbox_inches='tight')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description="Train a model.") 
    parser.add_argument('--mode', type=str, default='none',help='options: dp, ddp, mp-train, mp-debug, none') 
    args = parser.parse_args() 

    # Training Parameters
    batch_size = 20  
    num_epochs = 1 
    lr = 0.001 
    momentum = 0.9 
    print_every_n = 5 

    model_name = 'inception-v3' 
    model = get_pretrained_model(model_name) 

    if args.mode == 'none': 
        print('Training without GPU acceleration!\n')
        exec_start = time.time() 
        model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=print_every_n) 
        print('--------------') 
        print(f'Total execution time without GPU acceleration: {time.time() - exec_start}') 

    else: 
        if torch.cuda.device_count() > 1:  
            print(f'Using {torch.cuda.device_count()} GPUs') 
            
            # Using DataParallel   
            if args.mode == 'dp':  
                print('Using DataParallel\n') 
                exec_start = time.time() 
                model = nn.DataParallel(model)  
                model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=print_every_n)  
                print('--------------') 
                print(f'Total DataParallel execution time: {time.time() - exec_start} seconds')

            # Using DistributedDataParallel 
            elif args.mode == 'ddp':
                print('Using DistributedDataParallel\n')  
                exec_start = time.time() 
                world_size = torch.cuda.device_count()  
                mp.spawn(
                    ddp_train, 
                    args = (world_size, model, model_name, batch_size, num_epochs, lr, momentum, print_every_n), 
                    nprocs = world_size, 
                    join=True
                )    
                print('---------------')
                print(f'Total DDP execution time: {time.time() - exec_start} seconds')  
            
            # Model Parallelism 
            elif args.mode == 'mp-debug': 
                get_comms_time_model_parallel()  
                
            elif args.mode == 'mp-train': 
                model = ModelParallelVGG16_4GPU(debugging=False) 
                exec_start = time.time() 
                model = train(model, 'vgg16', batch_size=32,print_every_n=print_every_n, model_parallel=True, num_epochs=1) 
                print()
                print('--------------') 
                print(f'Total execution time Model Parallelism: {time.time() - exec_start}')
    