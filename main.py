import argparse 
import time 
import torch 
import torch.nn as nn 
import torch.multiprocessing as mp  
from preprocess import get_pretrained_model   
from train import train, ddp_train 


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
        

        
    else: 
        print('Training without GPU acceleration!\n')
        model = train(model,model_name,batch_size=batch_size, num_epochs=1, print_every_n=10) 
