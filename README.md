# Autoencoder-in-Pytorch
Implement Convolutional Autoencoder in PyTorch with CUDA   
The Autoencoders, a variant of the artificial neural networks, are applied in the image process especially to reconstruct the images.
The image reconstruction aims at generating a new set of images similar to the original input images.  

### Autoencoder  
To demonstrate the use of convolution transpose operations, we will build an autoencoder.  

An autoencoder is not used for supervised learning. We will no longer try to predict something about our input. Instead, an autoencoder is considered a generative model:  
it learns a distributed representation of our training data, and can even be used to generate new instances of the training data..  

An autoencoder model contains two components:  

***An encoder that takes an image as input, and outputs a low-dimensional embedding (representation) of the image.  
A decoder that takes the low-dimensional embedding, and reconstructs the image.***  

![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/autoencoder-model.png)  

### Convolutional Autoencoder  
Convolutional Autoencoder is a variant of Convolutional Neural Networks that are used as the tools for unsupervised learning of convolution filters.
They are generally applied in the task of image reconstruction to minimize reconstruction errors by learning the optimal filters they can be applied to any input in order to extract features. Convolutional Autoencoders are general-purpose feature extractors differently from general autoencoders that completely ignore the 2D image structure. In autoencoders, the image must be unrolled into a single vector and the network must be built following the constraint on the number of inputs. 
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/structure-Convolutional-AutoEncoders.png)
### Variational autoencoders.
### stacked autoencoders

### Sparse autoencoders
### Denoising autoencoders
### Principal components analysis (PCA) VS autoencoders
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/pca-AE.png)

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
#### Import Data
```
train_img = []
for img_name in tqdm(train['Image_Name']):
    # defining the image path
    image_path =  str(img_name) + '.JPG'
    
    # reading the image
    img = imread(image_path, as_gray=True )
    img = resize(img, (img.shape[0] // 9, img.shape[1] // 9),anti_aliasing=True)
                      
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)
```
crate Validation 
```
# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
```
converting training images into torch format
```
# converting training images into torch format
train_x = train_x.reshape(81, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape
```
converting validation images into torch format
```
# converting validation images into torch format
val_x=val_x.reshape(21, 1, 28, 28)
val_x=torch.from_numpy(val_x)


# converting the target into torch format
val_y=val_y.astype(int);
val_y=torch.from_numpy(val_y)


# shape of validation data
val_x.shape, val_y.shape
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
![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/U-Net-model.png)  

![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/AE-segmentation.png)  

![image](https://github.com/E008001/Autoencoder-in-Pytorch/blob/main/CNN-AE.jpg)  
