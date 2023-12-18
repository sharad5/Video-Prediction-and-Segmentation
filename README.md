# Video Prediction with SimVP

### Instructions to run

1. Train the SimVP Model
`python FramePrediction/SimVP/main.py --data_root=/scratch/sd5251/DL/Project/clevrer1/dataset --res_dir=FramePrediction/SimVP/results --disable_wandb=true`
Arguments: 
- `data_root` - Root Folder for the dataset
- `res_dir` - Folder to Store the trained model checkpoints

2. Run Inference on the trained SimVP Model
`python FramePrediction/SimVP/main.py --inference=true --inference_data_root="/scratch/sd5251/DL/Project/clevrer1/dataset/val" --inference_file_name="simvp_basic_val_set_wobbly-snow-10_53.pt" --inference_model_checkpoint=/scratch/sn3250/DL/project/Video-Prediction-and-Segmentation/FramePrediction/SimVP/results/Debug/checkpoints/wobbly-snow-10_53.pth.pth --res_dir=FramePrediction/SimVP/results`
Arguments:
- `inference` - Boolean flag to run inference
- `inference_data_root` - Root Folder for the inference dataset
- `inference_file_name` - Filename for the predictions to be saved in
- `inference_model_checkpoint` - Model checkpoint tot use to run inference
- `res_dir` - Folder to Store the inference in

3. Train the Segmentation Model (UNet)
`python SemanticSegmentation/UNet/main.py --train_data_dir=/scratch/sd5251/DL/Project/clevrer1/dataset/train --val_data_dir=/scratch/sd5251/DL/Project/clevrer1/dataset/val --res_dir=SemanticSegmentation/UNet/results`
Arguments: 
- `train_data_dir` - Root Folder for the train dataset
- `val_data_dir` - Root Folder for the validation dataset
- `res_dir` - Folder to Store the trained model checkpoints

4. Run Inference on the trained UNet Model
`python SemanticSegmentation/UNet/main.py --inference=true --inference_frames_file="/scratch/sd5251/DL/Project/Video-Prediction-and-Segmentation/FramePrediction/SimVP/results/inference/simvp_basic_val_set_wobbly-snow-10_53.pt" --res_dir=SemanticSegmentation/UNet/results --inference_file_name=simvp_basic_val_set_wobbly-snow-10_53_floral-deluge-72_11.pt --inference_model_checkpoint=/scratch/sd5251/DL/Project/Video-Prediction-and-Segmentation/SemanticSegmentation/UNet/results/checkpoints/floral-deluge-72_11.pth`
Arguments:
- `inference` - Boolean flag to run inference
- `inference_frames_file` - Path to the file where SimVP Inferences are stores
- `inference_file_name` - Filename for the predictions to be saved in
- `inference_model_checkpoint` - Model checkpoint tot use to run inference
- `res_dir` - Folder to Store the inference 

5. Evaluate on Val Set
`python SemanticSegmentation/UNet/evaluate.py --prediction_file=SemanticSegmentation/UNet/results/inference/simvp_basic_val_set_wobbly-snow-10_49_floral-deluge-72_11.pt --eval_data_dir=/scratch/sd5251/DL/Project/clevrer1/dataset/val --res_dir=SemanticSegmentation/UNet/results`
Arguments:
- `prediction_file` - File with the predicted masks
- `eval_data_dir` - Data Directory for the Validation Set
- `inference_file_name` - Filename for the predictions to be saved in
- `inference_model_checkpoint` - Model checkpoint tot use to run inference
- `res_dir` - Folder where the true frames will be stored for later use 