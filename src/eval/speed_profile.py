import torch
import time

def measure_latency(model, input_sample, device, num_runs=100):
    """
    Measure inference latency.
    """
    model.eval()
    model.to(device)
    input_sample = input_sample.to(device)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_sample)
            end = time.time()
            times.append(end - start)
    return np.mean(times) * 1000  # ms

def count_mflops(model, input_sample):
    """
    Count MFLOPS.
    """
    from torchprofile import profile_macs
    macs = profile_macs(model, input_sample)
    return macs / 1e6  # MFLOPS
