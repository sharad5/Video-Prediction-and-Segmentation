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

def image_filename_2_idx(filename):
    return int(filename.split("_")[1].split(".")[0])

class ClevrerSegmentationTrainDataSet(Dataset):
    def __init__(self, video_directory, keep_last_only=False, transform=None):
        self.transforms = transform
        self.images = self._prepare_images(video_directory, keep_last_only)

    def _prepare_images(self, video_directory, keep_last_only):
        images = []
        for directory in video_directory:
            image_files = [f for f in os.listdir(directory) if not "mask" in f]
            sorted_images = sorted(image_files, key=extract_image_num)
            image_paths = [os.path.join(directory, img) for img in sorted_images]

            if keep_last_only:
                images.append(image_paths[-1])
            else:
                images.extend(image_paths)

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        image_filename = self.images[index].split('/')[-1]
        image_idx = image_filename_2_idx(image_filename)
        video_path = '/'.join(self.images[index].split('/')[:-1])
        mask = np.load(video_path + '/mask.npy')[image_idx, :, :]

        if self.transforms is not None:
            # Only Image Transformations like Blur
            img = self.transforms(img)

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