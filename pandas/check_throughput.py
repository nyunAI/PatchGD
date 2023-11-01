import torch
import time
from torchvision.models import resnet50
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
from prettytable import PrettyTable
os.environ["CUDA_VISIBLE_DEVICES"] = '2'



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class PatchDataset(Dataset):
    def __init__(self,images,num_patches,stride,patch_size):
        self.images = images
        self.num_patches = num_patches
        self.stride = stride
        self.patch_size = patch_size
    def __len__(self):
        return self.num_patches ** 2
    def __getitem__(self,choice):
        i = choice%self.num_patches
        j = choice//self.num_patches
        return self.images[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size], choice

class Backbone(nn.Module):
    def __init__(self,baseline,latent_dim):
        super(Backbone,self).__init__()
        # self.encoder = timm.create_model('resnet10t',pretrained=True)
        self.encoder = resnet50(pretrained=True)
        if baseline:
            self.encoder.fc = nn.Linear(2048,NUM_CLASSES)
        else:
            self.encoder.fc = nn.Linear(2048,latent_dim)
    def forward(self,x):
        return self.encoder(x)
 
class CNN_Block(nn.Module):
    def __init__(self,latent_dim,num_classes,num_patches):
        super(CNN_Block,self).__init__()
        self.expected_dim = (2,latent_dim,num_patches,num_patches)
        self.layer1 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,1,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.dropout = nn.Dropout2d(p=0.2)
        flatten_dim = self.get_final_out_dimension(self.expected_dim)
        self.linear = nn.Linear(flatten_dim,num_classes)

    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    def get_final_out_dimension(self,shape):
        s = shape
        s = self.get_output_shape(self.layer1,s)
        s = self.get_output_shape(self.layer2,s)
        s = self.get_output_shape(self.layer3,s)
        s = self.get_output_shape(self.layer4,s)
        return np.prod(list(s[1:]))

    def forward(self,x,print_shape=False):
        x = self.layer1(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer2(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer3(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer4(x)
        if print_shape:
            print(x.size())
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)
        if print_shape:
            print(x.size())
        return x



@torch.no_grad()
def compute_baseline_throughput(batch_size,image_size):
    # torch.cuda.empty_cache()
    model = Backbone(True,0)
    count_parameters(model)
    model.eval()
    model.to(ACCELARATOR)
    timing = []

    inputs = torch.rand(batch_size,3,image_size,image_size)
    inputs = inputs.to(ACCELARATOR)

    # warmup
    for _ in range(WARMUP):
        model(inputs)

    torch.cuda.synchronize()
    for _ in range(NUM_ITERATIONS):
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    return batch_size / timing.mean()



@torch.no_grad()
def compute_ultracnn_throughput(batch_size,
                                latent_dimension,
                                image_size,
                                num_classes,
                                num_patches,
                                stride,
                                patch_size,
                                percent_sampling):
    torch.cuda.empty_cache()
    model1 = Backbone(False,latent_dimension)
    model2 = CNN_Block(latent_dimension,num_classes,num_patches)

    p = count_parameters(model1)
    p += count_parameters(model2)

    print(f'Overall params:{p}')

    model1.to(ACCELARATOR)
    model2.to(ACCELARATOR)

    model1.eval()
    model2.eval()

    timing = []

    inputs = torch.rand(batch_size,3,image_size,image_size)
    inputs = inputs.to(ACCELARATOR)

    # warmup
    for _ in range(WARMUP):
        patch_dataset = PatchDataset(inputs,num_patches,stride,patch_size)
        patch_loader = DataLoader(patch_dataset,int(len(patch_dataset)*percent_sampling),shuffle=True)
        
        L1 = torch.zeros((batch_size,latent_dimension,num_patches,num_patches))
        L1 = L1.to(ACCELARATOR)

        for patches, idxs in patch_loader:
            patches = patches.to(ACCELARATOR)
            patches = patches.reshape(-1,3,patch_size,patch_size)
            out = model1(patches)
            out = out.reshape(-1,batch_size, latent_dimension)
            out = out.permute((1,2,0))
            row_idx = idxs//num_patches
            col_idx = idxs%num_patches
            L1[:,:,row_idx,col_idx] = out

        model2(L1)
    

    torch.cuda.synchronize()
    for _ in range(NUM_ITERATIONS):
        start = time.time()
        
        patch_dataset = PatchDataset(inputs,num_patches,stride,patch_size)
        patch_loader = DataLoader(patch_dataset,int(len(patch_dataset)*percent_sampling),shuffle=True)
        
        L1 = torch.zeros((batch_size,latent_dimension,num_patches,num_patches))
        L1 = L1.to(ACCELARATOR)

        for patches, idxs in patch_loader:
            patches = patches.to(ACCELARATOR)
            patches = patches.reshape(-1,3,patch_size,patch_size)
            out = model1(patches)
            out = out.reshape(-1,batch_size, latent_dimension)
            out = out.permute((1,2,0))
            row_idx = idxs//num_patches
            col_idx = idxs%num_patches
            L1[:,:,row_idx,col_idx] = out

        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    return batch_size / timing.mean()

if __name__ == '__main__':
    ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    IMAGE_SIZES = [512,2048,4096]
    BATCH_SIZES_BASELINE = [560,35,8]
    BATCH_PATCH_GD = [1700,197,55]

    BATCH_SIZES_BASELINE = [1,1,1]
    BATCH_PATCH_GD = [1,1,1]

    SAMPLING = [0.3,0.1,0.1]
    

    WARMUP = 50
    NUM_CLASSES = 6
    NUM_ITERATIONS = 1
    LATENT_DIMENSION = 256
    PATCH_SIZE = 128
    STRIDE = PATCH_SIZE

    for img_size,batch_size in zip(IMAGE_SIZES,BATCH_SIZES_BASELINE):
        print(f'Image size: {img_size}, batch size: {batch_size}')
        images_per_sec = compute_baseline_throughput(batch_size,img_size)
        print(f"Baseline at {batch_size} batch size: {images_per_sec:.2f}")



    for img_size,batch_size,sampling in zip(IMAGE_SIZES,BATCH_PATCH_GD,SAMPLING):
        num_patches = ((img_size-PATCH_SIZE)//STRIDE) + 1
        print(f'Image size: {img_size}, batch size: {batch_size}, #patches: {num_patches}, sampling: {sampling}')
        images_per_sec = compute_ultracnn_throughput(batch_size=batch_size,
                            latent_dimension=LATENT_DIMENSION,
                            image_size=img_size,
                            num_classes=NUM_CLASSES,
                            num_patches=num_patches,
                            stride=STRIDE,
                            patch_size=PATCH_SIZE,
                            percent_sampling=sampling)
        print(f"Ultracnn with {sampling*100} % sampling at {batch_size} batch size: {images_per_sec:.2f}")
