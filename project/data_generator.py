import os
import cv2
import random
import string
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torchvision.datasets as datasets
import pandas as pd
 
 
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
 
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
 
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
 
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
 
    if x_right < x_left or y_bottom < y_top:
        return 0.0
 
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
 
    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
   
    return iou
 
 
class CreateUltraMNIST:
    def __init__(self, root: str, base_data_path: str, n_samples: list = [28000, 28000],
                    img_size: int = 4000, img_scale_fact: list = [1, 76])->None:
        self.root_path = root
        self.base_data_path = base_data_path
        self.n_samples = n_samples
        self.img_size = img_size
        self.img_scale_fact = img_scale_fact
        self.n_classes = 28
        self.data = None
        self.targets = None
 
        # check if data exists
        self.data_exists_flag = False
        self.download_base_flag = True
        self.sum_list = defaultdict(list)
        self.dataframe = []
 
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    result = [i, j, k]
                    result.sort()
                    if result not in self.sum_list[sum(result)]:
                        self.sum_list[sum(result)].append(result)
                    for u in range(10):
                        result = [i, j, k, u]
                        result.sort()
                        if sum(result)<=self.n_classes and result not in self.sum_list[sum(result)]: self.sum_list[sum(result)].append(result)
                        for v in range(10):
                            result = [i, j, k, u, v]
                            result.sort()
                            if sum(result)<=self.n_classes and result not in self.sum_list[sum(result)]: self.sum_list[sum(result)].append(result)
 
    def generate_dataset(self):
        if self.data_exists_flag:
            raise Exception('Data already exists, delete the content to download again')
 
        print('Checking for base dataset, if needed')
        self.get_base_dataset()
 
        print('Preparing storage locations')
        os.mkdir(self.root_path)
 
        # creating train test and validation folders
        os.mkdir(os.path.join(self.root_path, 'train'))
        os.mkdir(os.path.join(self.root_path, 'val'))
 
        # limiting the sample per class (spc)
        train_spc = int(self.n_samples[0] / self.n_classes)
        val_spc = int(self.n_samples[1] / self.n_classes)
 
        # generating samples
        self._generate_samples(os.path.join(self.root_path, 'train'), train_spc)
        df = pd.DataFrame(self.dataframe, columns=['image_id','digit_sum'])
        df.to_csv(f"{self.root_path}/train.csv",index=False)
        self._generate_samples(os.path.join(self.root_path, 'val'), val_spc)
 
    def _generate_samples(self, data_path, spc):
        # spc denotes samples per class
        for num_class in range(self.n_classes):
            combinations = self.sum_list[num_class]
            for i in tqdm(range(spc)):
                labels = combinations[np.random.choice(len(combinations))]
                images = [self.data[self.targets==label][np.random.choice(len(self.data[self.targets==label]))] for label in labels]
 
                # generate sample
                img, label = self._generate_one_sample(images, labels)
 
                img_dir = data_path
 
                letters = string.ascii_lowercase
                fname = ''.join(random.choice(letters) for j in range(10))
                im = Image.fromarray(img*255)
                im.convert('L').save(os.path.join(img_dir, fname+'.jpeg'))
                self.dataframe.append([fname,label])
 
    def _generate_one_sample(self, images, labels):
        # creating the background
        img = np.zeros((self.img_size, self.img_size))
 
        label = 0
        prev_boxes = []
 
        # Add scaled versions of base image into the main image at random locations
        i = 0
        while i < len(images):
            sub_img = images[i]
 
            # random sample a resoltion from V-shape distribution
            k = int(np.ceil((self.img_scale_fact[1]-self.img_scale_fact[0])/2))
            prob = np.array([i for i in range(k, 0, -1)] + [i for i in range(1, k)])
            res_fact = np.random.choice(range(self.img_scale_fact[0], self.img_scale_fact[1]), p=prob/prob.sum())
 
            if res_fact == 1 and np.random.rand()<0.5:
                scaled_simg = sub_img.numpy()
                scaled_simg = cv2.resize(scaled_simg, (14, 14), interpolation=cv2.INTER_NEAREST)
            else: scaled_simg = np.kron(sub_img, np.ones((res_fact,res_fact)))
 
            # add to img
            sub_len = scaled_simg.shape[0]
            randx = random.randint(0, img.shape[0]-sub_len)
            randy = random.randint(0, img.shape[0]-sub_len)
 
            # add to prev_boxes, if overlap with all boxes in prev_boxes is less
            new_box = {}
            new_box = {'x1': randx, 'x2': randx+sub_len-1, 'y1': randy, 'y2': randy+sub_len-1}
            add_flag = self._check_for_low_overlap(new_box, prev_boxes)
 
            if add_flag:
                img[randx:randx+sub_len, randy:randy+sub_len] += scaled_simg
                prev_boxes.append(new_box)
                # updating the label
                label += labels[i]
                i += 1
 
        img[img > 1] = 1
        return img, label
 
    def get_base_dataset(self):
        # check if base dataset exists, else download it
        if not os.path.exists(self.base_data_path):
            print('Base dataset does not exist at specified path, downloading now...')
            self.download_base_flag = True
 
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])
 
        if self.download_base_flag:            
            mnist_trainset = datasets.MNIST(root=self.base_data_path, train=True, download=True, transform=transform)
        else:
            mnist_trainset = datasets.MNIST(root=self.base_data_path, train=True, download=False, transform=transform)
 
        self.data = mnist_trainset.data/255.
        self.targets = mnist_trainset.targets
 
    def _check_for_low_overlap(self, new_box, prev_boxes):
        # if prev_boxes is empty, add this box, so return True
        if not prev_boxes:
            return True
 
        # if there is atleast one element in prev_boxes
        add_flag = True
        for box in prev_boxes:
            iou = get_iou(new_box, box)
            if iou > 0:
                add_flag = False
 
        return add_flag
 
 
 
if __name__ == '__main__':
 
    ROOT_DIRECTORY = './ultramnist_sample'
    BASE_DATASET = './mnist'
    SAMPLES_PER_CLASS = 1000
    N_SAMPLES = SAMPLES_PER_CLASS * np.array([28,28])
    IMAGE_SIZE = 4000
 
    obj_umnist = CreateUltraMNIST(root=ROOT_DIRECTORY,
                                    base_data_path=BASE_DATASET,
                                    n_samples = N_SAMPLES,
                                    img_size=IMAGE_SIZE)
    obj_umnist.generate_dataset()

