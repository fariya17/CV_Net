# CV-Net: Attention-Augmented LSTM for Real-Time Driving Behavior and Road Anomaly Detection

## Project Summary

CV-Net is an attention-augmented Long Short-Term Memory (LSTM) framework designed for real-time detection of driving behaviors and road anomalies using smartphone sensor data. The model leverages calibration-aware preprocessing to mitigate sensor biases, a two-stage LSTM architecture for temporal feature extraction, and additive attention mechanisms to focus on critical temporal segments. This enables high-accuracy classification with minimal latency, making it suitable for embedded systems in vehicles.

## Key Contributions

- **Calibration-Aware Preprocessing**: Implements bias drift compensation and uncertainty quantification to ensure sensor reliability.
- **Two-Stage LSTM Architecture**: Utilizes stacked LSTM layers with dropout for robust temporal modeling.
- **Additive Attention Mechanism**: Enhances interpretability by weighting important time steps.
- **Real-Time Performance**: Achieves 99.18% accuracy with 0.118 ms latency on edge devices.

## Dataset Details

The dataset includes six driving behaviors (e.g., normal driving, aggressive acceleration) and 17 road anomaly types (e.g., potholes, speed bumps), sampled at approximately 90 Hz (range: 60–99 Hz). It is available on Figshare under the title “Harnessing Smartphone Sensors for Enhanced Road Safety.”

## Reproduction Guide

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Preprocess Data**: `python scripts/preprocess.py`
3. **Train Model**: `python scripts/train.py --config configs/cvnet.yaml`
4. **Evaluate**: `python scripts/evaluate.py`
5. **Generate Figures/Tables**: `python scripts/feature_importance.py` and `python scripts/export_model.py`

## Results Snapshot

### Table XI (Ablation Study)
- S1: Baseline (no BN, no attention) - Accuracy: 95.2%
- S2: +Batch Norm - Accuracy: 96.8%
- S3: +Attention - Accuracy: 97.5%
- S4: +Dense Units - Accuracy: 98.1%
- S5: +Dropout - Accuracy: 98.7%
- S6: +Multi-LSTM - Accuracy: 99.0%
- S7-S10: Variants with full features - S10: 99.18%

### Table XII (Comparison)
- CV-Net: 99.18% Accuracy, 0.118 ms Latency
- LSTM Baseline: 94.5% Accuracy, 0.152 ms Latency
- CNN: 96.2% Accuracy, 0.189 ms Latency
- Transformer: 98.0% Accuracy, 0.245 ms Latency

## Citation

Please cite our paper if you use this code:

```
@article{author2023cvnet,
  title={CV-Net: Attention-Augmented LSTM for Real-Time Driving Behavior and Road Anomaly Detection},
  author={Author et al.},
  journal={Journal},
  year={2023}
}
```

Contact: author@example.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
