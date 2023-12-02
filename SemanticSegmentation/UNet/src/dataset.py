import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import logging

class ClevrerSegmentationDataSet(Dataset):
    def __init__(self, video_dir, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in video_dir:
            imgs = os.listdir(i)
            imgs_in_video_dir = [i + '/' + img for img in imgs if not img.startswith('mask')]
            self.images.extend(np.random.choice(imgs_in_video_dir, 11))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        x = self.images[index].split('/')
        image_name = x[-1]
        mask_index = int(image_name.split("_")[1].split(".")[0])
        x = x[:-1]
        mask_path = '/'.join(x)
        mask = np.load(mask_path + '/mask.npy')
        mask = mask[mask_index, :, :]

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask

def get_dataloaders(args):
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    batch_size = args.batch_size
    
    train_data_dir = [train_data_dir + f"/video_{i}" for i in range(0, 10)]
    train_dataset = ClevrerSegmentationDataSet(train_data_dir, None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_data_dir = [val_data_dir + f"/video_{i}" for i in range(1000, 1010)]
    val_dataset = ClevrerSegmentationDataSet(val_data_dir, None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader