import torch
from BlendedNeRF.run_BlendedNeRF import train_blended_nerf

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"Number of GPUs:{torch.cuda.device_count()}")
    train_blended_nerf()