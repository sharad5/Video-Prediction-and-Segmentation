import os
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
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
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
            self.model.load_state_dict(torch.load(self.args.inference_model_checkpoint))
        
        if torch.cuda.device_count() >= 1:
            self.model = torch.nn.DataParallel(self.model) #, device_ids=None will take in all available devices
            print(f"Using {torch.cuda.device_count()} GPUs!")
        
        self.model.to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        config["data_root"] += "/hidden"
        config["inference"] = True
        self.hidden_loader, self.data_mean, self.data_std = load_data(**config)
        #self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
#             if i * batch_x.shape[0] > 1000:
#                 break
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))
            
            #pred_y_norm = pred_y / 255.0
            #batch_y_norm = batch_y / 255.0
                
            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        folder_path = self.path+'/results/inference/preds/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, 0, 1, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        for np_data in ['trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])

        print({'vali_mse': mse, 'vali_mae': mae, 'vali_ssim': ssim, 'vali_psnr': psnr})
        return total_loss

