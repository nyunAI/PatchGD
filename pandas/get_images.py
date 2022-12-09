import skimage.io
from glob import glob
from tqdm import tqdm
import os
import cv2

def get_best_resolution_index(io_object,dimension):
    best_index = len(io_object) - 1
    for i in reversed(range(len(io_object))):
        shape = io_object[i].shape
        if dimension > min(shape[0],shape[1]):
            break
        best_index = i
    return best_index

def save_images(root_dir,save_dir_root,dimensions):    
    for path in tqdm(glob(root_dir+'\\*')):
        image_id = path.split('\\')[-1].split('.')[0]
        biopsy = skimage.io.MultiImage(path)
        best_index = get_best_resolution_index(biopsy,max(*dimensions))
        for dim in dimensions:
            im = biopsy[best_index]
            old_size = im.shape[:2] # old_size is in (height, width) format
            ratio = float(dim)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            # new_size should be in (width, height) format
            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = dim - new_size[1]
            delta_h = dim - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [255, 255, 255]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

            save_dir = f"{save_dir_root}\\training_images_{dim}"
            cv2.imwrite(save_dir + '\\' + image_id+'.png', new_im)

def make_directories(root_dir,dimensions):
    for dim in dimensions:
        dir_name = f"{root_dir}\\training_images_{dim}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


if __name__ == '__main__':
    ROOT_DIR = '..\\data\\prostate-cancer-grade-assessment\\train_images'
    SAVE_DIR = '..\\data\\pandas_dataset'
    DESIRED_DIMENSIONS = [512,1024,2048,4096]
    make_directories(SAVE_DIR,DESIRED_DIMENSIONS)
    save_images(ROOT_DIR,SAVE_DIR,DESIRED_DIMENSIONS)
