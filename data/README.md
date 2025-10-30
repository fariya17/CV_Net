# Dataset Documentation

## Overview

This directory contains the dataset for the CV-Net project, focusing on driving behavior and road anomaly detection using smartphone sensors.

## Dataset Source

The dataset is available on Figshare: [“Harnessing Smartphone Sensors for Enhanced Road Safety”](https://figshare.com/).

## Directory Structure

```
data/
├── raw/                # Original CSV sensor data per drive
├── processed/          # Preprocessed numpy arrays
└── metadata/           # Calibration and session details
```

## Sensor Description

- **Accelerometer**: Measures linear acceleration (x, y, z axes).
- **Gyroscope**: Measures rotational velocity (x, y, z axes).
- **Gravity**: Gravity vector (x, y, z axes).
- **Linear Acceleration**: Acceleration without gravity (x, y, z axes).
- **Orientation**: Device orientation (quaternion or Euler angles).
- **GPS**: Location and speed data.

## Sampling Frequency

Approximately 90 Hz (range: 60–99 Hz).

## Ethics and Privacy

All data is anonymized with no personally identifiable information (PII). Participation was voluntary, and ethical guidelines were followed as stated in the manuscript.

## Data Split Summary

- **Training**: 70%
- **Test**: 20%
- **Validation**: 10%

## Processed Data Contents

Generated after running `scripts/preprocess.py` (Algorithm 1):

- `X_train.npy`, `X_val.npy`, `X_test.npy`: 3D arrays of shape [samples, 50, 15] (0.5 s windows × 15 features).
- `y_train.npy`, `y_val.npy`, `y_test.npy`: Class labels.
- `folds.json`: 5-fold indices for cross-validation.
- `feature_names.json`: List of sensor channels used.
- `stats.json`: Mean/variance for normalization.

## Metadata Contents

- `devices.json`: Smartphone models, IMU specifications, mounting details.
- `calibration_log.csv`: Time-stamped offset/drift measurements.
- `sessions.csv`: Each driving session, environment (urban/highway), speed range, road type.
- `reference_instruments.csv`: Comparison logs for calibration reproducibility.
