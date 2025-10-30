#!/usr/bin/env python3
"""
Evaluation script.
"""
from src.eval.evaluation import evaluate_model, save_confusion_matrix, save_metrics
from src.training.metrics import compute_metrics

# Load model and data
# preds, labels = evaluate_model(model, test_loader, device)
# metrics = compute_metrics(labels, preds)
# save_confusion_matrix(labels, preds, 'results/figures/confusion.png')
# save_metrics(metrics, 'results/tables/metrics.csv')

print("Evaluation completed.")
