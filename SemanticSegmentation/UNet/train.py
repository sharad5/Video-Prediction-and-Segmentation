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
from src.unet import UNet
from src.utils import check_accuracy

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
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)

    # Model Parameters
    parser.add_argument('--use_model_checkpoint', default=False, type=bool)
    parser.add_argument('--model_checkpoint_file', default='checkpoint.pth', type=str)

    return parser



if __name__ == "__main__":
    # hyperparameters

    max_patience = 3
    epochs_no_improve = 0
    early_stop = False
    SMOOTH = 1e-6
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = create_parser().parse_args()
    cfg = args.__dict__

    experiment = wandb.init(project='unet-seg', config=cfg)#, mode="disabled")
    
    train_dataloader, val_dataloader = get_dataloaders(args)
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda')
    print('Use GPU {}:'.format(args.gpu))

    model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)
#     model = model.to(memory_format=torch.channels_last)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
#     model.load_state_dict(torch.load(unet_model_saved_path).state_dict(),strict=False)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # , device_ids=None will take in all available devices
        print(f"Using {torch.cuda.device_count()} GPUs!")

    model.to(device)
    best_model = None

    # loss criterion, optimizer

    weights = [0.00132979]+[1]*48 # Downweighting the class 0 because it is ~94% of the classes
    class_weights = torch.FloatTensor(weights).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # Train loop
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader)
        train_epoch_loss = 0
        for idx, (data, targets) in enumerate(loop):
            data = data.permute(0, 3, 1, 2).to(torch.float16).to(device)
            targets = targets.to(device)
            targets = targets.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            train_epoch_loss += loss.item()
        
        wandb.log(
            {
                f'train_epoch_loss': train_epoch_loss/len(train_dataloader)
            }
        )
        val_losses = 0
        last_val_loss = 1000000
        model.eval()
        mean_thresholded_iou = []
        ious = 0
        last_iou = 0
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(device)
                y = y.to(device)
                y = y.type(torch.long)
                # forward
                with torch.cuda.amp.autocast():
                    preds = model(x)
                    vloss = loss_fn(preds, y)

                val_losses += vloss.item()
                
                class_labels = {j: "object_" + str(j) if j != 0 else "background" for j in range(49)}
                pred_mask_for_log = torch.argmax(preds, dim=1)[0].cpu().detach().numpy()
                true_mask_for_log = y[0].cpu().detach().numpy()
                if (i+1)%50 == 0:
                    wandb.log(
                       {f"image_{i}" : wandb.Image(x[0], masks={
                               f'pred_masks':
                                   {"mask_data":pred_mask_for_log, "class_labels":class_labels},
                              f'true_masks':
                                   {"mask_data":true_mask_for_log, "class_labels":class_labels}
                           })
                       })
                preds_arg = torch.argmax(softmax(preds), axis=1)

                thresholded_iou = batch_iou_pytorch(SMOOTH, preds_arg, y)
                ious += thresholded_iou
                
            wandb.log({f'val_loss_epoch':val_losses/len(val_dataloader)})
            mean_thresholded_iou = ious / len(val_dataloader)
            avg_val_loss = val_losses / len(val_dataloader)
            print(f"Epoch: {epoch}, avg IoU: {mean_thresholded_iou}, avg val loss: {avg_val_loss}")

        if avg_val_loss < last_val_loss:
            best_model = model
            torch.save(best_model, f'checkpoints/{experiment.name}_best_model.pt')
            last_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > max_patience and epoch > 10:
            early_stop = True
            print("Early Stopping")

    check_accuracy(val_dataloader, best_model)

# if __name__=='__main__':
#    sweep_configuration = {
#           "method": "grid",
#           "parameters":{
#                "learning_rate": {"values":[1e-4]}
#        }
#    }

# Start the sweep
#    sweep_id = wandb.sweep(
#        sweep=sweep_configuration,
#        project='unet-seg',
#        )

#    wandb.agent(sweep_id, function=main, count=1)
