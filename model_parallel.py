import torch 
import torch.nn as nn 
from preprocess import get_pretrained_model

class ModelParallelVGG16_4GPU(nn.Module): 
    '''
    Split VGG16 layers across 4 GPU devices
    '''
    def __init__(self, debugging=False): 
        super(ModelParallelVGG16_4GPU, self).__init__() 
        model = get_pretrained_model('vgg16')   
        self.overhead = 0  
        self.debugging = debugging

        self.seq1 = model.features[:len(model.features)//2].to('cuda:0') 
        self.seq2 = model.features[len(model.features)//2:].to('cuda:1') 
        self.avgpool = model.avgpool.to('cuda:2') 
        self.classifier = model.classifier.to('cuda:3')  
    
    def forward(self, x): 
        if self.debugging: 
            return self.forward_debug(x) 
        
        else: 
            return self.forward_normal(x) 

    def forward_normal(self, x): 
        x = x.to('cuda:0')  
        out = self.seq1(x)  
        out = out.to('cuda:1') 
        out = self.seq2(out)
        out = out.to('cuda:2') 
        out = self.avgpool(out) 
        out = out.to('cuda:3') 
        out = self.classifier(out) 

        return out  

    def forward_debug(self, x): 

        # Initialize CUDA Event Listeners to measure time taken for the data transfer to specific GPUs 
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)  

        # When doing CUDA operations, all operations get added to a stream and are executed in order. 
        # CUDA operations are asynchronous by default so they don't wait for previous ones to finish before the next one starts however which can mess up the time measurements for specific operations that we are interested in. 
        # Synchronize makes sure all the gpu operations in the stream are first completed. 

        # Ignore first one because this would be done even for the base case without Model Parallelism 
        x = x.to('cuda:0')  
        out = self.seq1(x)  

        start_time.record()  
        out = out.to('cuda:1') 
        end_time.record() 
        torch.cuda.synchronize()   
        self.overhead += start_time.elapsed_time(end_time) 
        
        out = self.seq2(out)

        start_time.record()
        out = out.to('cuda:2') 
        end_time.record() 
        torch.cuda.synchronize()
        self.overhead += start_time.elapsed_time(end_time)  

        out = self.avgpool(out) 
        
        start_time.record()
        out = out.to('cuda:3') 
        end_time.record()
        torch.cuda.synchronize()
        self.overhead += start_time.elapsed_time(end_time)

        return self.overhead         
    

class ModelParallelVGG16_3GPU(nn.Module): 
    '''
    Split VGG16 layers across 3 GPU devices
    '''
    def __init__(self, debugging=False): 
        super(ModelParallelVGG16_3GPU, self).__init__() 
        model = get_pretrained_model('vgg16')  
        self.overhead = 0 
        self.debugging = debugging

        self.seq1 = model.features.to('cuda:0') 
        self.avgpool = model.avgpool.to('cuda:1') 
        self.classifier = model.classifier.to('cuda:2')  

    def forward(self, x): 
        if self.debugging: 
            return self.forward_debug(x) 
        else: 
            return self.forward_normal(x) 

    def forward_normal(self, x): 
        x = x.to('cuda:0')  
        out = self.seq1(x)  
        out = out.to('cuda:1') 
        out = self.avgpool(out) 
        out = out.to('cuda:2') 
        out = self.classifier(out) 

        return out   
    
    def forward_debug(self,x): 
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)  

        x = x.to('cuda:0')  
        out = self.seq1(x)  

        start_time.record()
        out = out.to('cuda:1') 
        end_time.record() 
        torch.cuda.synchronize()
        self.overhead += start_time.elapsed_time(end_time)  

        out = self.avgpool(out) 
        
        start_time.record()
        out = out.to('cuda:2') 
        end_time.record()
        torch.cuda.synchronize()
        self.overhead += start_time.elapsed_time(end_time)

        return self.overhead    


class ModelParallelVGG16_2GPU(nn.Module): 
    '''
    Split VGG16 layers across 2 GPU devices
    '''
    def __init__(self, debugging=False): 
        super(ModelParallelVGG16_2GPU, self).__init__() 
        model = get_pretrained_model('vgg16')   
        self.overhead = 0 
        self.debugging = debugging 

        self.seq1 = model.features.to('cuda:0') 
        self.avgpool = model.avgpool.to('cuda:1') 
        self.classifier = model.classifier.to('cuda:1') 

    def forward(self, x): 
        if self.debugging: 
            return self.forward_debug(x) 
        
        else: 
            return self.forward_normal(x) 

    def forward_normal(self, x): 
        x = x.to('cuda:0')  
        out = self.seq1(x)  
        out = out.to('cuda:1') 
        out = self.avgpool(out) 
        out = self.classifier(out) 
        return out  

    def forward_debug(self,x): 
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)  

        x = x.to('cuda:0')  
        out = self.seq1(x)  

        start_time.record()
        out = out.to('cuda:1') 
        end_time.record() 
        torch.cuda.synchronize()
        self.overhead += start_time.elapsed_time(end_time)  

        return self.overhead  
