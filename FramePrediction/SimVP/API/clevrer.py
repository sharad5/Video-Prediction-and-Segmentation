import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch.utils.data import random_split, Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_video_number(s):
    return int(s.split('_')[1])

def extract_image_number(s):
    return int(s.split('_')[1].split('.')[0])

class ClevrerTrainDataSet(Dataset):
    def __init__(self, root_directory, is_training=True, input_frame_count=11, output_frame_count=11, transform=None):
        super(ClevrerTrainDataSet, self).__init__()
        self.video_directories = self._load_video_directories(root_directory)
        self.dataset_size = len(self.video_directories)
        self.is_training = is_training
        self.input_frame_count = input_frame_count
        self.output_frame_count = output_frame_count
        self.transformation = transform

    def _load_video_directories(self, root_directory):
        directory_names = sorted(os.listdir(root_directory), key=extract_video_number)
        return [os.path.join(root_directory, dir_name, '') for dir_name in directory_names]

    def _load_video_images(self, directory_path, frame_count):
        image_filenames = sorted([file for file in os.listdir(directory_path) if file.endswith('.png')], key=extract_image_number)
        images = [np.array(Image.open(os.path.join(directory_path, filename))) for filename in image_filenames[:frame_count]]
        return images

    def __getitem__(self, index):
        directory_path = self.video_directories[index]
        initial_frames = self._load_video_images(directory_path, self.input_frame_count)
        subsequent_frames = self._load_video_images(directory_path, self.output_frame_count)

        tensor_initial_frames = torch.stack([torch.from_numpy(frame) for frame in initial_frames]).permute(0, 3, 1, 2)
        tensor_subsequent_frames = torch.stack([torch.from_numpy(frame) for frame in subsequent_frames]).permute(0, 3, 1, 2)

        return tensor_initial_frames.contiguous().float(), tensor_subsequent_frames.contiguous().float()

    def __len__(self):
        return self.dataset_size


class ClevrerInferenceDataSet(Dataset):
    def __init__(self, root_directory, input_frame_count=11, transform=None):
        super(ClevrerInferenceDataSet, self).__init__()
        self.video_directories = self._load_video_directories(root_directory)
        self.dataset_size = len(self.video_directories)
        self.input_frame_count = input_frame_count
        self.transformation = transform

    def _load_video_directories(self, root_directory):
        directory_names = sorted(os.listdir(root_directory), key=extract_video_number)
        return [os.path.join(root_directory, dir_name, '') for dir_name in directory_names]

    def _load_video_images(self, directory_path, frame_count):
        image_filenames = sorted([file for file in os.listdir(directory_path) if file.endswith('.png')], key=extract_image_number)
        images = [np.array(Image.open(os.path.join(directory_path, filename))) for filename in image_filenames[:frame_count]]
        return images

    def __getitem__(self, index):
        directory_path = self.video_directories[index]
        initial_frames = self._load_video_images(directory_path, self.input_frame_count)
        tensor_initial_frames = torch.stack([torch.from_numpy(frame) for frame in initial_frames]).permute(0, 3, 1, 2)

        return tensor_initial_frames.contiguous().float()

    def __len__(self):
        return self.dataset_size

def load_clevrer_inference_data(batch_size, data_root, num_workers):
    inference_dataset = ClevrerInferenceDataSet(root_directory=data_root, input_frame_count=11)
    inference_data_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return inference_data_loader

def load_clevrer(batch_size, val_batch_size, data_root, num_workers):
    training_dataset = ClevrerTrainDataSet(root_directory=data_root + "/unlabeled", is_training=True, input_frame_count=11, output_frame_count=11)
    validation_dataset = ClevrerTrainDataSet(root_directory=data_root + "/val", is_training=False, input_frame_count=11, output_frame_count=11)

    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataset_mean, dataset_std = 0, 1
    return training_data_loader, validation_data_loader, dataset_mean, dataset_std