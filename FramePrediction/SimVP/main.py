import argparse
from exp import Exp
from inference import Inference

import warnings
warnings.filterwarnings('ignore')
import wandb

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--disable_wandb', default=False, type=bool)
    
    # Inference Parameters
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--inference_model_checkpoint', default="results/Debug/checkpoint.pth", type=str)
    parser.add_argument('--inference_data_root', default="/scratch/sd5251/DL/Project/clevrer1/dataset/hidden", type=str)
    parser.add_argument('--inference_file_name', default="simvp_basic_hidden_set.pt", type=str)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/scratch/sd5251/DL/Project/clevrer1/dataset/unlabeled')
    parser.add_argument('--dataname', default='clevrer', choices=['clevrer'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[11,3,160,240], type=int,nargs='*')
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default="one-cycle", type=str, help='Learning rate Scheduler')
    parser.add_argument('--use_weighted_mse', default=False, type=bool, help='MSE Weighted for frames of interest')
    parser.add_argument('--use_model_checkpoint', default=False, type=bool)
    parser.add_argument('--model_checkpoint_file', default="results/Debug/checkpoints_old/19.pth", type=str)
    return parser



if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    print(config)
    if args.inference:
        print('Starting Inference')
        inf = Inference(args)
        inf.run()
        print('Inference Completed')
    else:
        print('Starting Training')
        wandb_params = {"project": "video-pred-simvp", "config": config}
        if args.disable_wandb:
            wandb_params["mode"] = "disabled"
        wandb_exp = wandb.init(**wandb_params)

        exp = Exp(args, wandb.config, wandb_exp.name)
        exp.train(args)
        wandb.finish()
        print('Training Completed')

