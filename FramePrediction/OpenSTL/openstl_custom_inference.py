import os
import cv2
import wandb
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from openstl.api import BaseExperiment
from openstl.utils import create_parser

batch_size = 4
pre_seq_length = 11 # x-frames
aft_seq_length = 11 # y-frames (to predict)


def extract_num(s):
    return int(s.split('_')[1])

def extract_image_num(s):
    return int(s.split('_')[1].split('.')[0])

class ClevrerDataSet(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11, transform=None):
        super(ClevrerDataSet, self).__init__()

        self.videos = []
        unlabelled_dirs = os.listdir(root)
        unlabelled_dirs = sorted(unlabelled_dirs, key=extract_num)

        for video in unlabelled_dirs:
            self.videos.extend([root + '/' + video + '/'])
        
        self.length = len(self.videos)

        self.is_train = is_train

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.mean = 0
        self.std = 1

    def __getitem__(self, index):
        length = self.n_frames_input + self.n_frames_output
        # print(self.videos[index])
        video_folder = os.listdir(self.videos[index])
        
        # Remove mask file (for train videos)
        if "mask.npy" in video_folder:
            video_folder.remove("mask.npy")
    
        video_folder = sorted(video_folder, key=extract_image_num)
        imgs = []
        for image in video_folder:
            imgs.append(np.array(Image.open(self.videos[index] + '/' + image)))

        #shape: torch.Size([10, 1, 64, 64])
        # print(len(imgs))

        past_clips = imgs[0:self.n_frames_input] #[11,160,240,3]
        future_clips = imgs[-self.n_frames_output:] #[11,160,240,3]

        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        future_clips = [torch.from_numpy(clip) for clip in future_clips]
        # stack the tensors and permute the dimensions

        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2)
        future_clips = torch.stack(future_clips).permute(0, 3, 1, 2)
        #we want [11,3,160,240]
        return (past_clips).contiguous().float(), (future_clips).contiguous().float()
    
    def __len__(self):
        return self.length


class ClevrerInferenceDataSet(Dataset):
    def __init__(self, root, n_frames_input=11, transform=None):
        super(ClevrerInferenceDataSet, self).__init__()

        self.videos = []
        unlabelled_dirs = os.listdir(root)
        unlabelled_dirs = sorted(unlabelled_dirs, key=extract_num)

        for video in unlabelled_dirs:
            self.videos.extend([root + '/' + video + '/'])
        
        self.length = len(self.videos)
        self.n_frames_input = n_frames_input
        self.transform = transform

    def __getitem__(self, index):
        length = self.n_frames_input
        # print(self.videos[index])
        video_folder = [file for file in os.listdir(self.videos[index]) if file != "mask.npy"]
        # print(video_folder)
        video_folder = sorted(video_folder, key=extract_image_num)
        imgs = []
        for image in video_folder:
            imgs.append(np.array(Image.open(self.videos[index] + '/' + image)))

        past_clips = imgs[0:self.n_frames_input] #[11,160,240,3]
        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2) # [11,3,160,240]
        return (past_clips).contiguous().float()
    
    def __len__(self):
        return self.length


def load_clevrer_inference_data(batch_size, data_root, num_workers):
    data = ClevrerInferenceDataSet(root=data_root, n_frames_input=11)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader


# dataloader_test = load_clevrer_inference_data(batch_size, data_root="../dataset/val/", 
                                                # num_workers=1)
dataset_test = ClevrerDataSet(root="../dataset/val/", is_train=False, 
                                n_frames_input=11, n_frames_output=11)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
print("Loaded Data")


# # SimVP2
# custom_model_config = {
#     'method': 'SimVP',
#     'model_type': 'gSTA',
#     'N_S': 4,
#     'N_T': 8,
#     'hid_S': 64,
#     'hid_T': 256,
#     "sched": "onecycle"
# }

# # SimVP + Swin Transformer
# custom_model_config = {
#                     "method": 'SimVP',
#                     "model_type": 'swin',
#                     "spatio_kernel_enc": 3,
#                     "spatio_kernel_dec": 3,
#                     "hid_S": 64,
#                     "hid_T": 512,
#                     "N_S": 4,                
#                     "N_T": 8,                    
#                     "drop_path": 0,
#                     "sched": 'onecycle'
#                     }

# PredRNNv2
custom_model_config = {
                    "method": 'PredRNNv2',
                    "reverse_scheduled_sampling": 1,
                    "r_sampling_step_1": 25000,
                    "r_sampling_step_2": 50000,
                    "r_exp_alpha": 5000,
                    "scheduled_sampling": 1,
                    "sampling_stop_iter": 50000,
                    "sampling_start_value": 1.0,
                    "sampling_changing_rate": 0.00002,
                    "num_hidden": '128,128,128,128',
                    "filter_size": 5,
                    "stride": 1,
                    "patch_size": 4,
                    "layer_norm": 0,
                    "decouple_beta": 0.1,
                    "sched": 'onecycle'
                }

# # SimVP + ViT
# custom_model_config = {
#                     "method": 'SimVP',
#                     "model_type": 'vit',
#                     "spatio_kernel_enc": 3,
#                     "spatio_kernel_dec": 3,
#                     "hid_S": 64,
#                     "hid_T": 512,
#                     "N_T": 8,
#                     "N_S": 4,
#                     "drop_path": 0,
#                     "sched": 'onecycle'
#                     }

custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 20, # 10
    'lr': 1e-3, # 5e-4
    'metrics': ['mse', 'mae'],
    'ex_name': 'custom_exp',
    'dataname': 'custom',
    'in_shape': [11, 3, 160, 240], # (T,C,H,W)
}
model_paths = {"res_dir": "predrnnv2_work_dirs/", "ex_name": "custom_exp"}

args = create_parser().parse_args([])
config = args.__dict__
config.update(custom_training_config) # update training config
config.update(custom_model_config) # update model config
config.update(model_paths)

#config['dist'] = True
#config['launcher'] = 'pytorch'
#config['local_rank'] = '0,1' # Multi-GPU training
#os.environ['RANK'] = '0,1'


#wandb.init(project="frame-pred-vit-simvp", entity="a-is-all-we-need", config=config)
exp = BaseExperiment(args, wandb.config, dataloaders=(dataloader_test, dataloader_test, dataloader_test))
print("Initialized Model")

print('>'*35 + ' Running Inference ' + '<'*35)
exp.inference()