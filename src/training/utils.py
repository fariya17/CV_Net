import torch
import random
import numpy as np

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_logger(name):
    """
    Get logger for training.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

def get_scheduler(optimizer, scheduler_type, patience):
    """
    Get learning rate scheduler.
    """
    if scheduler_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
