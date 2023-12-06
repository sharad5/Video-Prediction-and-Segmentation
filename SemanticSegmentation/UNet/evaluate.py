import os
import torch
import torchmetrics
from tqdm import tqdm
import argparse

from src.dataset import get_eval_dataset

def create_parser():
    parser = argparse.ArgumentParser()
    # Evaluation Parameters 
    parser.add_argument('--prediction_file', default="./results/inference/simvp_basic_val_set_legendary-forest-67_18.pt", type=str)
    parser.add_argument('--eval_data_dir', default='/scratch/sd5251/DL/Project/clevrer1/dataset/val', type=str)
    return parser



if __name__ == "__main__":    
    args = create_parser().parse_args()
    cfg = args.__dict__
    print(cfg)
    eval_dataset = get_eval_dataset(args)
    true_last_frames = []
    # print(eval_dataset.shape)
    for _, mask in tqdm(eval_dataset):
        # print(video[0].shape)
        true_last_frames.append(torch.tensor(mask))
    pred_last_frames = torch.load(args.prediction_file)
    true_last_frames = torch.stack(true_last_frames, dim=0).to(pred_last_frames.device)
    torch.save(true_last_frames, "./results/inference/true_last_frames.pt")
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(pred_last_frames.device)

    jac_idx = jaccard(true_last_frames, pred_last_frames)
    print(f"Jaccard Index: {jac_idx}")
