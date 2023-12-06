import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import logging

def extract_image_num(s):
    return int(s.split('_')[-1].split('.')[0])

class ClevrerSegmentationTrainDataSet(Dataset):
    def __init__(self, video_dir, keep_last_only=False, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in video_dir:
            imgs = os.listdir(i)
            imgs_in_video_dir = [i + '/' + img for img in imgs if not img.startswith('mask')]
            imgs_in_video_dir = sorted(imgs_in_video_dir, key=extract_image_num)
            # print(imgs_in_video_dir)
            if keep_last_only:
                self.images.append(imgs_in_video_dir[-1])
            else:
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

class ClevrerSegmentationFrameInferenceDataSet(Dataset):
    def __init__(self, inference_frames_file, frames_to_predict, transform=None):
        self.transforms = transform
        self.data = torch.load(inference_frames_file)#[:,frames_to_predict,...]
        self.data = self.data[:, frames_to_predict, :, :, :]
        num_videos, num_frames_per_video, num_channels, height, width = self.data.shape
        self.frames = self.data.view(num_videos * num_frames_per_video, num_channels, height, width)

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, index):
        frame = self.frames[index, ...]

        if self.transforms is not None:
            aug = self.transforms(image=img)
            frame = aug['image']

        return frame

def get_frame_inference_dataloader(args):
    inference_frames_file = args.inference_frames_file
    batch_size = args.inference_batch_size
    
    inference_dataset = ClevrerSegmentationFrameInferenceDataSet(inference_frames_file, frames_to_predict=[-1], transform=None)
    inference_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    return inference_dataloader

def get_eval_dataset(args):
    eval_data_dir = args.eval_data_dir
    eval_data_dir = [eval_data_dir + f"/video_{i}" for i in range(1000, 2000)]
    eval_dataset = ClevrerSegmentationTrainDataSet(eval_data_dir, keep_last_only=True, transform=None)
    return eval_dataset

def get_dataloaders(args):
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    batch_size = args.batch_size
    
    train_data_dir = [train_data_dir + f"/video_{i}" for i in range(0, 1000)]
    train_dataset = ClevrerSegmentationTrainDataSet(train_data_dir, None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_data_dir = [val_data_dir + f"/video_{i}" for i in range(1000, 2000)]
    val_dataset = ClevrerSegmentationTrainDataSet(val_data_dir, None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader