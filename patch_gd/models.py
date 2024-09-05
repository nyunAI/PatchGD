import torch.nn as nn
from torchvision.models import resnet50
from config import model_config
import numpy as np
import torch



class Backbone(nn.Module):
    def __init__(self,latent_dim):
        super(Backbone,self).__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048,latent_dim)
    def forward(self,x):
        return self.encoder(x)

class CNN_Block(nn.Module):
    def __init__(self,num_layers_sr,latent_dim,feature_dim,num_classes,num_patches):
        super(CNN_Block,self).__init__()
        self.expected_dim = (2,latent_dim,num_patches,num_patches)
        self.same_layers,self.reduction_layers = num_layers_sr #(same,reduction)
        self.layers = []
        self.feature_dim = feature_dim
        for _ in range(self.same_layers):
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(latent_dim,self.feature_dim,3,1,1), 
                nn.ReLU(),
                nn.BatchNorm2d(self.feature_dim),
                nn.Dropout2d(p=0.2))
            )
        d = self.feature_dim
        if self.same_layers == 0:
            d = latent_dim
        for _ in range(self.reduction_layers):
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(d,self.feature_dim,3,2,1), 
                nn.ReLU(),
                nn.BatchNorm2d(self.feature_dim),
                nn.Dropout2d(p=0.2))
            )
        self.layers = nn.Sequential(*self.layers)
        flatten_dim = self.get_final_out_dimension(self.expected_dim)
        self.linear = nn.Linear(flatten_dim,num_classes)

    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    def get_final_out_dimension(self,shape):
        s = shape
        s = self.get_output_shape(self.layers,s)
        return np.prod(list(s[1:]))

    def forward(self,x):
        for l in self.layers:
            x = l(x)
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)
        return x



def get_head(config_type,latent_dimension):
    num_layers_sr = (1,3)
    feaure_dimension = 256
    num_classes = 6
    num_patches = 16
    
    if config_type == model_config.SMALLER:
        num_layers_sr = (0,3)
    elif config_type == model_config.LARGER:
        num_layers_sr = (1,4)
    elif config_type == model_config.SMALLER_FEAT:
        feaure_dimension = 128
    elif config_type == model_config.LARGER_FEAT:
        feaure_dimension = 512
    return CNN_Block(num_layers_sr=num_layers_sr,
                    latent_dim=latent_dimension,
                    feature_dim=feaure_dimension,
                    num_classes=num_classes,
                    num_patches=num_patches)
    

def get_head_from_name(name,latent_dimension):
    configurations = {'original':model_config.ORIGINAL,
                    'smaller': model_config.SMALLER,
                    'larger': model_config.LARGER,
                    'smaller_feat': model_config.SMALLER_FEAT,
                    'larger_feat': model_config.LARGER_FEAT}
    return get_head(configurations[name],latent_dimension)

