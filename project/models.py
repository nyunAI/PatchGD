import torch.nn  as nn
from torchvision.models import resnet18
import timm
from constants import *
import numpy as np

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

class Backbone(nn.Module):
    def __init__(self,baseline=BASELINE):
        super(Backbone,self).__init__()
        self.encoder = timm.create_model('resnet10t',pretrained=True)
        # self.encoder = resnet18(pretrained=True)
        if baseline:
            self.encoder.fc = nn.Linear(512,NUM_CLASSES)
        else:
            self.encoder.fc = nn.Linear(512,LATENT_DIMENSION)
    def forward(self,x):
        return self.encoder(x)
 
class CNN_Block(nn.Module):
        def __init__(self,latent_dim=LATENT_DIMENSION,num_classes=NUM_CLASSES):
            super(CNN_Block,self).__init__()
            self.expected_dim = (BATCH_SIZE,latent_dim,NUM_PATCHES,NUM_PATCHES)
            self.layer1 = nn.Sequential(
                nn.Conv2d(latent_dim,128,3,2,2), 
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(128,64,3,2,2), 
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )
            layer_1_out_shape = get_output_shape(self.layer1,self.expected_dim)
            layer_2_out_shape = get_output_shape(self.layer2,layer_1_out_shape)
            flatten_dim = np.prod(list(layer_2_out_shape[1:]))

            self.linear = nn.Linear(flatten_dim,num_classes)

        def forward(self,x,print_shape=False):
            x = self.layer1(x)
            if print_shape:
                print(x.size())
            x = self.layer2(x)
            if print_shape:
                print(x.size())
            x = x.reshape(x.shape[0],-1)
            x = self.linear(x)
            if print_shape:
                print(x.size())
            return x
