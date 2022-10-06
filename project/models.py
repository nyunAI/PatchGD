import torch.nn  as nn
from torchvision.models import resnet18
from constants import *


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(512,256)
    def forward(self,x):
        return self.encoder(x)
 
class CNN_Block(nn.Module):
        def __init__(self,latent_dim=LATENT_DIMENSION,num_classes=NUM_CLASSES):
            super(CNN_Block,self).__init__()
            self.head = nn.Sequential(
                nn.Conv2d(latent_dim,256,(3,3),2), # 13 - > 6
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
                nn.Linear(128,num_classes)
            )
        def forward(self,x):
            return self.head(x)

