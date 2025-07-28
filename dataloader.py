import torch
import os
from PIL import Image
import random

def get_image_list(raw_image_path, clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    for raw_image in raw_image_list:
        image_file = raw_image.split('/')[-1]
        image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
    
    return image_list


class EUVPDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, is_train=False, prop=0.8):
        self.is_train = is_train
        self.image_list = get_image_list(data_path + 'Inp/', data_path + 'GTr/', True)
        self.transform = transform

        random.seed(42)
        random.shuffle(self.image_list)
        if is_train:
            self.image_list = self.image_list[:int(len(self.image_list) * prop)]
        else:
            self.image_list = self.image_list[int(len(self.image_list) * prop):]

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        clear_image = Image.open(clear_image)
        return self.transform(raw_image), self.transform(clear_image), image_name

    def __len__(self):
        return len(self.image_list)
    
class UIEBDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, is_train=False):
        self.is_train = is_train
        if is_train:
            self.image_list = get_image_list(data_path + 'raw/', data_path + 'reference/', is_train)
        else:
            self.image_list = get_image_list(data_path + 'raw_test/', data_path + 'reference_test/', True)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        clear_image = Image.open(clear_image)
        return self.transform(raw_image), self.transform(clear_image), image_name

    def __len__(self):
        return len(self.image_list)
