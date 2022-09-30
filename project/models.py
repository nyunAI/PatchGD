from tkinter.tix import IMAGE
import torch
import torch.optim as optim
import torch.nn  as nn
import timm
import numpy as np
import pytorch_lightning as pl
from constants import *
import torch.functional as F

class CustomModel(pl.LightningModule):
    def __init__(self, 
    latent_dim=LATENT_DIMENSION, 
    sampling_fraction=PERCENT_SAMPLING, 
    image_size=IMAGE_SIZE, 
    patch_size=PATCH_SIZE, 
    stride=STRIDE, 
    num_classes=NUM_CLASSES):
        super().__init__()
        # Constants:
        self.latent_dim = latent_dim
        self.sampling_fraction = sampling_fraction
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = ((self.image_size-self.patch_size)//self.stride) + 1
        self.num_classes = num_classes
        # Backbone and Head
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True) 
        self.backbone.classifier = nn.Linear(1280, self.latent_dim)
        self.head = nn.Sequential(
            nn.Conv2d(self.latent_dim,256,(3,3),2), # 13 - > 6
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(3,3), 1), # 6 - > 4 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(4,4), 1), #4 - > 1
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,self.num_classes)
        )
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
        self.criterion = nn.CrossEntropyLoss()
        

    def forward_no_grad_fill(self, images, L1):
        with torch.no_grad():
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = images[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size]
                    out = self.backbone(patch)
                    L1[:,:,i,j] = out
        return L1

    def forward(self, x, print_shape=False):
        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        with torch.no_grad():
            out = L1
            for layer in self.head:
                out = layer(out)
                if print_shape:
                    print(out.size())
        return out
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        
        patches = self.num_patches**2
        sampled = np.random.choice(patches, int(self.sampling_fraction*patches),replace=False)

        for choice in sampled:
            i = choice%self.num_patches
            j = choice//self.num_patches
            patch = x[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size]
            out = self.backbone(patch)
            L1[:,:,i,j] = out
        
        outputs = self.head(L1)
        loss = self.criterion(outputs,y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        outputs = self.head(L1)
        loss = self.criterion(outputs,y)
        self.log('val_loss', loss)

    
    def configure_optimizers(self):
        layer_names = ['head','backbone']
        lrs = {
            'head': LEARNING_RATE_HEAD,
            'backbone': LEARNING_RATE_BACKBONE
        }
        parameters = []
        for idx, name in enumerate(layer_names):
            parameters += [{'params': [p for n, p in self.named_parameters() if name == n.split('.')[0] and p.requires_grad],
                            'lr': lrs[name]}]
        optimizer = torch.optim.Adam(parameters)
        return optimizer

