import torch
import numpy as np

def evaluate_external(model, dataloader, device):
    """
    Evaluate on external datasets (UAH-DriveSet, Gonzalez).
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

def compute_generalization_error(preds, labels):
    """
    Compute cross-dataset generalization error.
    """
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(labels, preds)
