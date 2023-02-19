import torch.nn as nn
from torchvision.models import resnet50
from config import *
class Backbone(nn.Module):
    def __init__(self,num_classes):
        super(Backbone,self).__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048,num_classes)
    def forward(self,x):
        return self.encoder(x)
