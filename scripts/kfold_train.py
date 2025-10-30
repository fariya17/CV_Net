#!/usr/bin/env python3
"""
K-fold cross-validation training script.
"""
import yaml
import torch
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.cvnet import CVNet
from src.training.loop import train_model, train_epoch, validate_epoch
from src.training.utils import set_seed, get_scheduler
from src.training.losses import WeightedCrossEntropy
from src.training.metrics import compute_metrics
from src.data.loaders import SensorDataset
from src.data.splits import kfold_split
from src.eval.evaluation import evaluate_model
from src.viz.training_curves import plot_training_curves
from torch.utils.data import DataLoader

def main(config_path):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    # Load full training data
    try:
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
    except FileNotFoundError:
        print("Error: Training data files not found in data/processed/")
        sys.exit(1)
    
    n_folds = 5
    kf_splits = kfold_split(len(X_train), n_folds=n_folds, seed=config.get('seed', 42))
    
    # Store results for each fold
    fold_results = []
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kf_splits):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create datasets and loaders
        train_dataset = SensorDataset(X_fold_train, y_fold_train)
        val_dataset = SensorDataset(X_fold_val, y_fold_val)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Model
        model = CVNet(input_size=X_train.shape[1], **config['model'])
        model.to(device)
        
        # Optimizer, scheduler, criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = get_scheduler(optimizer, 'ReduceLROnPlateau', config.get('patience', 5))
        criterion = WeightedCrossEntropy()
        
        # Train
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        fold_val_accs = []
        
        best_loss = float('inf')
        patience_counter = 0
        patience = config.get('early_stop_patience', 10)
        
        for epoch in range(config['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            # Compute accuracies (simplified, assuming classification)
            model.eval()
            train_preds = []
            train_labels = []
            with torch.no_grad():
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    preds = torch.argmax(outputs, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(y.cpu().numpy())
            train_acc = accuracy_score(train_labels, train_preds)
            
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'experiments/kfold/fold_{fold+1}_best_model.pth')
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Evaluate on validation set
        val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
        fold_results.append(val_metrics)
        
        # Plot training curves for this fold
        plot_training_curves(fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs,
                            save_path=f'results/figures/kfold_fold_{fold+1}_training_curves.png')
        
        # Collect for overall plot
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_train_accs.append(fold_train_accs)
        all_val_accs.append(fold_val_accs)
    
    # Compute average metrics across folds
    avg_metrics = {}
    for key in fold_results[0].keys():
        avg_metrics[key] = np.mean([res[key] for res in fold_results])
    
    print("\n--- Average Validation Metrics Across Folds ---")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    os.makedirs('experiments/kfold', exist_ok=True)
    with open('experiments/kfold/kfold_results.yaml', 'w') as f:
        yaml.dump({'fold_results': fold_results, 'average_metrics': avg_metrics}, f)
    
    # Plot overall training curves (mean across folds)
    max_len = max(len(l) for l in all_train_losses)
    overall_train_losses = np.mean([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_train_losses], axis=0)
    overall_val_losses = np.mean([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_val_losses], axis=0)
    overall_train_accs = np.mean([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_train_accs], axis=0)
    overall_val_accs = np.mean([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in all_val_accs], axis=0)
    
    plot_training_curves(overall_train_losses, overall_val_losses, overall_train_accs, overall_val_accs,
                        save_path='results/figures/kfold_overall_training_curves.png')
    
    print("K-fold cross-validation completed. Results saved to experiments/kfold/")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python kfold_train.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
