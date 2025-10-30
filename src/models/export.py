import torch
import onnx

def export_to_onnx(model, input_sample, filepath):
    """
    Export PyTorch model to ONNX format.
    """
    torch.onnx.export(model, input_sample, filepath, opset_version=11)

def export_to_torchscript(model, filepath):
    """
    Export PyTorch model to TorchScript.
    """
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, filepath)
