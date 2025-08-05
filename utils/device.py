import os, torch
def get_device(prefer_gpu: bool = True):
    env = os.getenv("SOCIAL_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda:0" if prefer_gpu and torch.cuda.is_available() else "cpu")