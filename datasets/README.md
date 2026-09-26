# Datasets Directory

This directory hosts the datasets used for the FL-IoT-IDS benchmarks.

## CICIoT2023

* **Official Source:** [Canadian Institute for Cybersecurity (CIC) - CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
* **Citation:** E. C. P. Neto et al., *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment"*, Sensors, 2023.
* **Format:** Merged CSV containing 39 network flow features and 1 label column (`group_class_name`).

### File Placement
Place the merged CSV file at:
```text
datasets/CICIOT2023/merged_CICIOT2023_data.csv
```

> **Note:** Raw CSV files (>100MB) are excluded from Git tracking via `.gitignore`.
