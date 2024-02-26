import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import torchvision.models as models  
import re 

def val(model_path, val_loader): 
    model = '' 
    device = torch.device('cuda')  
    match = re.search(r'_(.*?)\.', model_path)
    model_name = match.group(1) if match else None

    if model_name == 'vgg16': 
        model = models.vgg16() 
        in_features = model.classifier[6].in_features 
        out_features = 10 

        for param in model.parameters():
            param.requires_grad = False 

        model.classifier[6] = nn.Linear(in_features=in_features, out_features=out_features, bias=True) 
        model.to(device) 

    elif model_name == 'inception-v3': 
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')  
        in_features = model.fc.in_features 
        out_features = 10 

        for param in model.parameters():
            param.requires_grad = False 

        model.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True) 
        model.to(device) 
    
    model.load_state_dict(torch.load(model_path)) 
    model.eval() 

    n_correct = 0 
    n_samples = 0
    

    with torch.no_grad(): 
        for features, targets in val_loader:  
            features = features.to(device) 
            targets = targets.to(device)  
            
            y_pred = model.forward(features)     
            _, predictions = torch.max(y_pred, 1) 

            n_correct += (predictions == targets).sum().item() 
            n_samples += targets.size(0) 
        
    
    return 100*(n_correct/n_samples) 
        
if __name__ == '__main__': 
    pass 

