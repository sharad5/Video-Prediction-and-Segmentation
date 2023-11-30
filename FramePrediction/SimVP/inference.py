import os
from collections import OrderedDict
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *

class Inference:
    def __init__(self, args):
        super(Inference, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()

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
        # log and checkpoint
        self.path = self.args.res_dir #osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()
    
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T)
        
        if os.path.isfile(self.args.inference_model_checkpoint):
            state_dict = torch.load(self.args.inference_model_checkpoint)
            if "module" in list(state_dict.keys())[0]:
                state_dict = OrderedDict({".".join(k.split(".")[1:]): v for k,v in state_dict.items()})
            self.model.load_state_dict(state_dict)
        
        if torch.cuda.device_count() >= 1:
            self.model = torch.nn.DataParallel(self.model) #, device_ids=None will take in all available devices
            print(f"Using {torch.cuda.device_count()} GPUs!")
        
        self.model.to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        config["data_root"] += "/hidden"
        config["inference"] = True
        self.data_loader = load_data(**config)

    def run(self):
        self.model.eval()
        predicted_batches = []
        pbar = tqdm(self.data_loader)
        with torch.no_grad():
            for i, batch_x in enumerate(pbar):
                batch_x = batch_x.to(self.device)
                pred_y = self.model(batch_x)
#                 print(pred_y.shape)
                predicted_batches.append(pred_y)
                if i==9:
                    break
        predictions = torch.cat(predicted_batches, dim=0)
        print(predictions.shape)
        folder_path = self.path+'/inference/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, self.args.inference_file_name)
        torch.save(predictions, file_path)
        print(f"Predictions saved to {file_path}")

