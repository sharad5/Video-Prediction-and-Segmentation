import os
from collections import OrderedDict
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from tqdm import tqdm

from src.dataset import get_frame_inference_dataloader
from src.unet import UNet
from src.utils import *

class Inference:
    def __init__(self, args):
        super(Inference, self).__init__()
        self.args = args
        # Logging Preperation
        self._preparation()
        # Set the device parameters
        self.device = self._acquire_device()
        # Prepare data
        self._get_data()
        # Build the model
        self._build_model()
        # Set the Automatic mixed precision
        # self._set_amp()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = torch.device('cuda')
            print_log('Use GPU {}:'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)

        self.path = self.args.res_dir #osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)
        # log and checkpoint
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')

    def _build_model(self):
        args = self.args
        self.model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)
        print(self.args.inference_model_checkpoint)

        if os.path.isfile(self.args.inference_model_checkpoint):
            state_dict = torch.load(self.args.inference_model_checkpoint)
            if "module" in list(state_dict.keys())[0]:
                state_dict = OrderedDict({".".join(k.split(".")[1:]): v for k,v in state_dict.items()})
            self.model.load_state_dict(state_dict)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model) #, device_ids=None will take in all available devices
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        
        self.model.to(self.device)

    def _get_data(self):
        self.data_loader = get_frame_inference_dataloader(self.args)
        self.class_labels = {j: "object_" + str(j) if j != 0 else "background" for j in range(self.args.num_classes)}

    def run(self):
        self.model.eval()
        predicted_batches = []
        pbar = tqdm(self.data_loader)
        with torch.no_grad():
            for i, batch_x in enumerate(pbar):
                # print(batch_x.shape)
                batch_x = batch_x.to(torch.float16).to(self.device)

                with torch.cuda.amp.autocast():
                    pred_y = self.model(batch_x)

                pred_y = torch.argmax(pred_y, dim=1)
                torch.cuda.empty_cache()
                predicted_batches.append(pred_y)
                # if i==9:
                #     break
        predictions = torch.cat(predicted_batches, dim=0)
        print(predictions.shape)
        folder_path = os.path.join(self.args.res_dir, 'inference')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, self.args.inference_file_name)
        torch.save(predictions, file_path)
        print_log(f"Predictions saved to {file_path}")

