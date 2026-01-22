# CV-Net: Attention-Augmented LSTM for Real-Time Driving Behavior and Road Anomaly Detection

## Project Summary

CV-Net is an attention-augmented Long Short-Term Memory (LSTM) framework designed for real-time detection of driving behaviors and road anomalies using smartphone sensor data. The model leverages calibration-aware preprocessing to mitigate sensor biases, a two-stage LSTM architecture for temporal feature extraction, and additive attention mechanisms to focus on critical temporal segments. This enables high-accuracy classification with minimal latency, making it suitable for embedded systems in vehicles.

## Key Contributions

- **Calibration-Aware Preprocessing**: Implements bias drift compensation and uncertainty quantification to ensure sensor reliability.
- **Two-Stage LSTM Architecture**: Utilizes stacked LSTM layers with dropout for robust temporal modeling.
- **Additive Attention Mechanism**: Enhances interpretability by weighting important time steps.

## Dataset Details

The dataset includes six driving behaviors (e.g., normal driving, aggressive acceleration) and 17 road anomaly types (e.g., potholes, speed bumps), sampled at approximately 90 Hz (range: 60–99 Hz). It is available on Figshare under the title “Harnessing Smartphone Sensors for Enhanced Road Safety.”


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
