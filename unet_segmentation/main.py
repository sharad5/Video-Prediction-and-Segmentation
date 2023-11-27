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
import wandb
import torchvision

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)

    return parser

class SegmentationDataSet(Dataset):

    def __init__(self, video_dir, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in video_dir:
            imgs = os.listdir(i)
            imgs_in_video_dir = [i + '/' + img for img in imgs if not img.startswith('mask')]
            self.images.extend(np.random.choice(imgs_in_video_dir, 2))
#             self.images.extend(imgs_in_video_dir)

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


class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(encoding_block, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Dropout(dropout))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=49, features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        dropout=0
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0], dropout)
        self.conv2 = encoding_block(features[0], features[1], dropout)
        self.conv3 = encoding_block(features[1], features[2], dropout)
        self.conv4 = encoding_block(features[2], features[3], dropout)
        self.conv5 = encoding_block(features[3] * 2, features[3], dropout)
        self.conv6 = encoding_block(features[3], features[2], dropout)
        self.conv7 = encoding_block(features[2], features[1], dropout)
        self.conv8 = encoding_block(features[1], features[0], 0)
        self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3], features[3] * 2, 0)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)
    # print(model)
    y_preds_list = []
    y_trues_list = []
    #ious = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            # print(x.shape)
            # plt.imshow(x.cpu()[0])
            x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(device)
            y = y.to(device)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)

            y_preds_list.append(preds)
            y_trues_list.append(y)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # thresholded_iou = batch_iou_pytorch(SMOOTH, preds, y)
            # ious.append(thresholded_iou)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # print(dice_score)
            # print(x.cpu()[0])
            # break

    # mean_thresholded_iou = sum(ious)/len(ious)

    y_preds_concat = torch.cat(y_preds_list, dim=0)
    y_trues_concat = torch.cat(y_trues_list, dim=0)
    print("IoU over val: ", mean_thresholded_iou)

    print(len(y_preds_list))
    print(y_preds_concat.shape)

    jac_idx = jaccard(y_trues_concat, y_preds_concat)

    print(f"Jaccard Index {jac_idx}")

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")


def batch_iou_pytorch(SMOOTH, outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # hyperparameters

    LEARNING_RATE = 5e-5
    weight_decay = 1e-3
    num_epochs = 15
    max_patience = 3
    epochs_no_improve = 0
    early_stop = False
    SMOOTH = 1e-6
    
    cfg = {
    "train": {"learning_rate":LEARNING_RATE, "epochs":num_epochs, "weight_decay":weight_decay}

    }

    wandb.init(project='unet-seg', config=cfg)
    args = create_parser().parse_args()

    train_set_path = '/scratch/sd5251/DL/Project/clevrer1/dataset/train/video_' #Change this to your train set path
    val_set_path = '/scratch/sd5251/DL/Project/clevrer1/dataset/val/video_' #Change this to your validation path
    unet_model_saved_path='./unet_10.pt'

    train_data_dir = [train_set_path + str(i) for i in range(0, 1000)]
    train_dataset = SegmentationDataSet(train_data_dir, None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_data_dir = [val_set_path + str(i) for i in range(1000, 2000)]
    val_dataset = SegmentationDataSet(val_data_dir, None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda')
    print('Use GPU {}:'.format(args.gpu))

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = unet_model()
    #model.load_state_dict(torch.load(unet_model_saved_path).state_dict(),strict=False)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # , device_ids=None will take in all available devices
        print(f"Using {torch.cuda.device_count()} GPUs!")

    model.to(device)
    best_model = None
    # summary(model, (3, 256, 256))

    

    # loss criterion, optimizer

#     loss_fn = nn.CrossEntropyLoss()
    weights = [0.00132979]+[1]*48 # Downweighting the class 0 because it is ~94% of the classes
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
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
                    preds = model(data)
                    vloss = loss_fn(preds, y)

#                 print(x.shape, y[0].shape, torch.argmax(preds, dim=1)[0].shape)
                # vloss = loss_fn(preds, y)
                val_losses += vloss.item()

                class_labels = {j: "object_" + str(j) if j != 0 else "background" for j in range(49)}
                pred_mask_for_log = torch.argmax(softmax(preds), dim=1)[0].cpu().detach().numpy()
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

                #thresholded_iou = batch_iou_pytorch(SMOOTH, preds_arg, y)
                #ious += thresholded_iou
                
            wandb.log({f'val_loss_epoch':val_losses/len(val_dataloader)})
            #mean_thresholded_iou = ious / len(val_dataloader)
            avg_val_loss = val_losses / len(val_dataloader)
            print(f"Epoch: {epoch}, avg val loss: {avg_val_loss}")

        if avg_val_loss < last_val_loss:
            best_model = model
            torch.save(best_model, '/scratch/cj2407/unet_15.pt')
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
