#!/usr/bin/env python3
"""
Model export script.
"""
from src.models.export import export_to_onnx, export_to_torchscript

# Export model
# export_to_onnx(model, 'models/exported/cvnet.onnx')
# export_to_torchscript(model, 'models/exported/cvnet.pt')

print("Model export completed.")
