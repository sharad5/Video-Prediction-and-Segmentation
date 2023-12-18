import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import argparse

import logging
import wandb

from src.dataset import get_dataloaders
from src import unet
from src.unet import UNet
from exp import Exp
from inference import Inference

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--amp_enabled', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--num_classes', default=49, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--train_data_dir', default='/scratch/sd5251/DL/Project/clevrer1/dataset/train', type=str)
    parser.add_argument('--val_data_dir', default='/scratch/sd5251/DL/Project/clevrer1/dataset/val', type=str)
    parser.add_argument('--dataname', default='clevrer', choices=['clevrer'])
    parser.add_argument('--num_workers', default=8, type=int)

    # Training parameters
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning Rate')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)

    # Inference Parameters 
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--inference_model_checkpoint', default="/scratch/sd5251/DL/Project/Video-Prediction-and-Segmentation/SemanticSegmentation/UNet/results/checkpoints/legendary-forest-67_18.pth", type=str)
    parser.add_argument('--inference_frames_file', default="/scratch/sd5251/DL/Project/Video-Prediction-and-Segmentation/FramePrediction/SimVP/results/inference/simvp_basic.pt", type=str)
    parser.add_argument('--inference_file_name', default="simvp_basic_worthy-breeze-62.pt", type=str)
    parser.add_argument('--inference_batch_size', default=24, type=int, help='Inference Batch size')

    # Model Parameters
    parser.add_argument('--use_model_checkpoint', default=False, type=bool)
    parser.add_argument('--model_checkpoint_file', default='checkpoint.pth', type=str)

    return parser



if __name__ == "__main__":    
    args = create_parser().parse_args()
    cfg = args.__dict__

    print(cfg)
    if args.inference:
        print('Starting Inference')
        inf = Inference(args)
        inf.run()
        print('Inference Completed')
    else:
        print('Starting Training')
        wandb_exp = wandb.init(project='unet-seg', config=cfg, mode="disabled")
        exp = Exp(args, wandb_exp.name)
        exp.train()
        wandb.finish()
        print('Training Completed')