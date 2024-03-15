The goal of this repository is for me to explore the different distributed training acceleration methods like data parallel and model parallel training. 

# Data Parallel 

I explore different types of data parallel techniques like DataParallel and DistributedDataParallel. In the future I also aim to implement RPC Framework to implement Parameter Server implementation for asynchronous data parallelization for boost in training speed. 

# Model Parallel 

For model parallelization, I make 4 variations of hosting different layers of a neural network architectures on different GPUs and show that choosing the right layer deployment is crucial. If the last layer hosted on one GPU has a very big sized output, the communication time for such a big output will be high and communication becomes a bottleneck. This has been shown through a graph comparing communication times between layers. 

  ## How do you measure communication time? 
  For doing this, I make use of `torch.cuda.Event()` to track start and end time before the x.to(gpu_device) operation. 

# Results 

I compare the execution times for passing data through all the different model and data parallel techniques, as well as the case without any distributed training. At the time of publishing this repo, I only experimented with a VGG16 model and was working with a GPU environment having 4x RTX3090. These were powerful GPUs and a single GPU here could easily handle a VGG16 model, which is why execution is fastest without any distributed training. 

This repo serves as my hands-on experience with setting up and executing distributed training environments in PyTorch. 
