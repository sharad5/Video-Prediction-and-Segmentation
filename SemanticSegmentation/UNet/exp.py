import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import logging
import wandb

from src.dataset import get_dataloaders
from src.unet import UNet
from src.utils import *

class Exp:
    def __init__(self, args, wandb_exp):
        super(Exp, self).__init__()
        self.args = args
        self.wandb_exp = wandb_exp
        # Logging Preperation
        self._preparation()
        # Set the device parameters
        self.device = self._acquire_device()
        # Prepare data
        self._get_data()
        # Build the model
        self._build_model()
        # Set the opimtizer
        self._select_optimizer()
        # Set the loss function
        self._select_criterion()
        # Set the Automatic mixed precision
        self._set_amp()
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = torch.device('cuda')
            self.logger.info('Use GPU {}:'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device
    
    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = self.args.res_dir
        check_dir(self.path)

        self.checkpoints_path = os.path.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filemode='a', format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info('Logging Setup!')

    def _build_model(self):
        args = self.args
        self.model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)
        model_checkpoint_path = os.path.join(self.path, self.args.model_checkpoint_file) # load saved model to restart from previous best model (lowest val loss) checkpoint

        if (self.args.use_model_checkpoint) and (os.path.isfile(model_checkpoint_path)):
            self.model.load_state_dict(torch.load(model_checkpoint_path))

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model) #, device_ids=None will take in all available devices
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        
        self.model.to(self.device)
    
    def _get_data(self):
        self.train_dataloader, self.val_dataloader = get_dataloaders(self.args)
        self.class_labels = {j: "object_" + str(j) if j != 0 else "background" for j in range(self.args.num_classes)}

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def _set_amp(self):
        if self.args.amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler()

    def _select_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _save(self, name=''):
        model_save_path = os.path.join(self.checkpoints_path, name + '.pth')
        torch.save(self.model.state_dict(), model_save_path)
    
    def _log_predicted_masks(self, key, image, pred_mask, true_mask):
        wandb.log(
                {
                    key : wandb.Image(image, masks={
                        f'pred_masks':
                            {"mask_data":pred_mask, "class_labels":self.class_labels},
                        f'true_masks':
                            {"mask_data":true_mask, "class_labels":self.class_labels}
                    })
                }
        )
    
    def train(self):
        
        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_dataloader)

            for idx, (batch_x, batch_y) in enumerate(train_pbar):
                # print(batch_x.shape)
                batch_x = batch_x.permute(0, 3, 1, 2).to(torch.float16).to(self.device)
                self.optimizer.zero_grad()
                batch_y = batch_y.to(torch.long).to(self.device)
                # pred_y = self.model(batch_x)
                with torch.cuda.amp.autocast():
                    pred_y = self.model(batch_x)
                    loss = self.criterion(pred_y, batch_y)
                
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # self.scheduler.step()
                torch.cuda.empty_cache()
            
            train_loss = np.average(train_loss)

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    val_loss, average_ious = self.val_eval()
                    torch.cuda.empty_cache()
                    #if epoch % (args.log_step * 100) == 0:
                    self._save(name=f"{self.wandb_exp}_{epoch}")
                self.logger.info("Epoch: {0} | Train Loss: {1:.4f} Val Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, val_loss))

                wandb.log({'Epoch Train Loss': train_loss, 'Epoch Val Loss': val_loss, 'Val Average IOU': average_ious})
        return self.model

    def val_eval(self):
        self.model.eval()
        average_ious, total_loss = [], []
        val_pbar = tqdm(self.val_dataloader)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_pbar):

                batch_x = batch_x.permute(0, 3, 1, 2).to(torch.float16).to(self.device)
                batch_y = batch_y.to(torch.long).to(self.device)
                # pred_y = self.model(batch_x)
                with torch.cuda.amp.autocast():
                    pred_y = self.model(batch_x)
                    loss = self.criterion(pred_y, batch_y)
                
                pred_batch_classes = torch.argmax(pred_y, dim=1)

                mask_logging_params = {
                    "key": f"image_{i}",
                    "image": batch_x[0],
                    "pred_mask": pred_batch_classes[0].cpu().detach().numpy(), 
                    "true_mask": batch_y[0].cpu().detach().numpy()
                }
                self._log_predicted_masks(**mask_logging_params)
                
                average_ious.append(average_iou(pred_batch_classes, batch_y).detach().cpu())
                total_loss.append(loss.item())
                val_pbar.set_description('val loss: {:.4f}'.format(loss.item()))
                
                torch.cuda.empty_cache()

        total_loss = np.average(total_loss)
        mean_average_iou = np.average(average_ious)
        return total_loss, mean_average_iou
