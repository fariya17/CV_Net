import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on test set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)

def save_confusion_matrix(y_true, y_pred, filepath):
    """
    Save confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def save_metrics(metrics, filepath):
    """
    Save metrics to CSV.
    """
    import pandas as pd
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
