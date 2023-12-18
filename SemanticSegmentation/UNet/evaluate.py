import os
import torch
import torchmetrics
from tqdm import tqdm
import argparse

from src.dataset import get_eval_dataset

def create_parser():
    parser = argparse.ArgumentParser()
    # Evaluation Parameters 
    parser.add_argument('--prediction_file', default="./results/inference/simvp_basic_val_set_wobbly-snow-10_49_floral-deluge-72_11.pt", type=str)
    parser.add_argument('--true_file', default="./results/inference/true_last_frames.pt", type=str)
    parser.add_argument('--eval_data_dir', default='/scratch/sd5251/DL/Project/clevrer1/dataset/val', type=str)
    parser.add_argument('--res_dir', default='./results', type=str)
    return parser



if __name__ == "__main__":
    args = create_parser().parse_args()
    cfg = args.__dict__
    print(cfg)
    eval_dataset = get_eval_dataset(args)
    true_last_frames = []
    true_imgs = []
    true_file = args.true_file
    if not os.path.isfile(args.true_file):
        for img, mask in tqdm(eval_dataset):
            true_last_frames.append(torch.tensor(mask))
            true_imgs.append(torch.tensor(img))

        folder_path = os.path.join(args.res_dir, 'inference')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        true_file = os.path.join(folder_path, "true_last_frames.pt")
        true_last_frames = torch.stack(true_last_frames, dim=0)
        torch.save(true_last_frames, true_file)

        # true_imgs = torch.stack(true_imgs, dim=0)
        # torch.save(true_imgs, "./results/inference/true_imgs.pt")
    
    pred_last_frames = torch.load(args.prediction_file)
    true_last_frames = torch.load(true_file).to(pred_last_frames.device)
    
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(pred_last_frames.device)

    jac_idx = jaccard(true_last_frames, pred_last_frames)
    print(f"Jaccard Index: {jac_idx}")
