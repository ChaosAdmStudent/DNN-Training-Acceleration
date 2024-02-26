import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import torchvision.models as models  
from preprocess import get_data 

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


def train(model:str, train_loader,print_every_n, dp=False, mp =False,num_epochs=5, lr = 0.001, momentum=0.9, model_save=False): 
    
    model_name = model  

    assert torch.cuda.is_available() 
    device = torch.device('cuda') 
    print(f'There are {torch.cuda.device_count()} GPUs available!')

    model = get_pretrained_model(model) 
    loss = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)  

    total_batches = len(train_loader)
    print('Starting training!') 
    n_correct = 0  
    n_samples = 0

    for i in range(1,num_epochs+1): 
        print_epoch = False 
        for j, (features, targets) in enumerate(train_loader):  
            features = features.to(device) 
            targets = targets.to(device)
            y_pred = model.forward(features) 

            if model_name == 'inception-v3': 
                y_pred, _ = y_pred

            l = loss(y_pred,targets)  
            _, predicted = torch.max(y_pred, 1) 
            n_correct += (predicted == targets).sum().item()  
            n_samples += targets.size(0)

            l.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

            if j % (total_batches//print_every_n) == 0: 
                if not print_epoch: 
                    print(f'Epoch {i}')
                    print_epoch = True 

                print(f'\t step {j}/{total_batches} loss: {l.item():.4f} acc: {100*(n_correct/n_samples):.2f}%')  
    
    print('Training complete!')
    
    if model_save: 
        torch.save(model.state_dict(), f'../model_{model_name}.pth')
    
    return model 


if __name__ == '__main__':  
    batch_size = 20 
    train_loader = get_data(batch_size=batch_size, train=True, download=True) 
    model = train('inception-v3', train_loader, num_epochs=1, print_every_n=10)  
    

