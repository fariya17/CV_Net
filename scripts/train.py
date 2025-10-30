#!/usr/bin/env python3
"""
Training script.
"""
import yaml
import torch
from src.models.cvnet import CVNet
from src.training.loop import train_model
from src.training.utils import set_seed, get_scheduler
from src.training.losses import WeightedCrossEntropy
from src.data.loaders import SensorDataset
from torch.utils.data import DataLoader

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    # Load data (placeholder)
    # X_train = np.load('data/processed/X_train.npy')
    # y_train = np.load('data/processed/y_train.npy')
    # train_dataset = SensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    
    # Model
    model = CVNet(input_size=15, **config['model'])
    
    # Optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = get_scheduler(optimizer, 'ReduceLROnPlateau', 5)
    criterion = WeightedCrossEntropy()
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, config['epochs'], config['early_stop_patience'], device)
    
    print("Training completed.")

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
