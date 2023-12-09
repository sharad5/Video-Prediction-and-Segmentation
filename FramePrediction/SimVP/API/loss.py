import torch
from torch import nn

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, num_frames, weight_scheme, device, custom_weights=None):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        if weight_scheme=="uniform":
            self.weights = torch.ones(num_frames)
        elif weight_scheme=="harmonic":
            self.weights = 1/(num_frames - torch.arange(num_frames))
        elif weight_scheme=="custom":
            if(len(custom_weights)!=num_frames):
                raise ValueError("Mismatch between length of custom weights and number of frames")
            self.weights = custom_weights
        self.weights = self.weights.to(device)
        # print(self.weights.device)
    
    def forward(self, input, target):
        unreduced_loss = self.criterion(input, target) # [B, 11, 3, H, W]
        partially_reduced_loss = unreduced_loss.mean(dim=(-3, -2, -1)) # [B, 11]
        # print(partially_reduced_loss.device)
        # print(self.weights.device)
        reduced_loss = (partially_reduced_loss * self.weights).mean()
        return reduced_loss
