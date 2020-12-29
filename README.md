# Autoencoder-in-Pytorch
Implement Convolutional Autoencoder in PyTorch with CUDA   
The Autoencoders, a variant of the artificial neural networks, are applied in the image process especially to reconstruct the images.
The image reconstruction aims at generating a new set of images similar to the original input images.  

### Autoencoder  
To demonstrate the use of convolution transpose operations, we will build an autoencoder.  

An autoencoder is not used for supervised learning. We will no longer try to predict something about our input. Instead, an autoencoder is considered a generative model:  
it learns a distributed representation of our training data, and can even be used to generate new instances of the training data.  

An autoencoder model contains two components:  

***An encoder that takes an image as input, and outputs a low-dimensional embedding (representation) of the image.  
A decoder that takes the low-dimensional embedding, and reconstructs the image.***

### Convolutional Autoencoder  
Convolutional Autoencoder is a variant of Convolutional Neural Networks that are used as the tools for unsupervised learning of convolution filters.
They are generally applied in the task of image reconstruction to minimize reconstruction errors by learning the optimal filters they can be applied to any input in order to extract features. Convolutional Autoencoders are general-purpose feature extractors differently from general autoencoders that completely ignore the 2D image structure. In autoencoders, the image must be unrolled into a single vector and the network must be built following the constraint on the number of inputs.  
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/structure-Convolutional-AutoEncoders.png)
### Variational autoencoders.
### stacked autoencoder

### Sparse autoencoders

```
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline
import torch.nn as nn
import torch.nn.functional as F
```
Model
```
# Writing our model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
### segmentation  
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/AE-segmentation.png)  
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/CNN-AE.jpg)  
