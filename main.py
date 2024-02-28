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
            print('Using DataParallel\n')
            model = nn.DataParallel(model)  
            model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=10) 

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
            print(f'Total execution time: {time.time() - exec_start} seconds')  
        
        # Model Parallelism 
        elif args.mode == 'mp': 
            print('Using Model Parallelism\n') 
            
            plt.figure(figsize=(10,6))
            data_loader_32 = get_dataloader(batch_size=32, train=True, is_dist=False) 
            data_loader_64 = get_dataloader(batch_size=64, train=True, is_dist=False) 

            labels = ['4 GPUs', '3 GPUs', '2 GPUs'] 
            x_pos = np.arange(len(labels))
            width = 0.3 
            overheads_32 = []  
            overheads_64 = [] 

            vgg16_4gpu = ModelParallelVGG16_4GPU(debugging=True)
            vgg16_3gpu = ModelParallelVGG16_3GPU(debugging=True)
            vgg16_2gpu = ModelParallelVGG16_2GPU(debugging=True) 

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

            # Plotting 
            plt.bar(x_pos - width/2, np.array(overheads_32), width, label='Batch_size = 32', color='black') 
            plt.bar(x_pos + width/2, np.array(overheads_64), width, label='Batch_size = 64', color='orange')
            plt.xticks(x_pos, labels) 
            plt.legend() 
            plt.show() 
            plt.savefig('../figures/model_parallel.png', bbox_inches='tight')

    
    else: 
        print('Training without GPU acceleration!\n')
        model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=10) 
