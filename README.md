# Network Anomaly Detection

![Created At](https://img.shields.io/github/created-at/rajkadakia/network-anomaly-detection)
![Contributors](https://img.shields.io/github/contributors/rajkadakia/network-anomaly-detection)
![License](https://img.shields.io/github/license/rajkadakia/network-anomaly-detection)

## Overview

This repository implements a **state-of-the-art network anomaly detection pipeline**, combining **classical machine learning** techniques with **large language models (LLMs)**. Designed for both **real-time** and **batch processing**, the system processes network logs, extracts relevant features, and detects a wide range of network attacks.

Built for **extensibility and scalability**, it utilizes the **CIC-IDS2017 dataset** for training, evaluation, and experimentation.

---

## Functional Architecture

The pipeline is structured into the following stages:
*(See the [functional diagram](https://github.com/rajkadakia/network-anomaly-detection/blob/main/docs/Functional_Diagram.jpg) for a visual representation)*

1. **Log Collection** – Aggregate logs from firewalls, routers, web servers, proxies, and other network devices.
2. **Log Parsing & Preprocessing** – Convert raw logs into structured formats suitable for analysis.
3. **Feature Extraction** – Extract key features from logs to serve as model inputs.
4. **Detection Engine** – Analyze features using a fine-tuned LLM (e.g., **FLAN-T5**) to classify events as **benign** or various **attack types**.
5. **Explanation Module** – Generate **human-readable alerts** and contextual information for detected anomalies.
6. **Model Training Pipeline** – Supports **training, fine-tuning, and evaluation** on labeled datasets.

---

## Dataset

* **Primary Dataset**: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), not included in the `datasets` directory due to size.
* **Preprocessed and Reduced Versions**: Available for rapid experimentation.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/rajkadakia/network-anomaly-detection.git
cd network-anomaly-detection
```

2. **Set up a Python virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Download the CIC-IDS2017 dataset**:

```bash
# Using wget
wget https://www.unb.ca/cic/datasets/machine-learning-ids-2017.html -O datasets/CIC-IDS2017.zip
unzip datasets/CIC-IDS2017.zip -d datasets/CIC-IDS2017

# Or using curl
curl -L https://www.unb.ca/cic/datasets/machine-learning-ids-2017.html -o datasets/CIC-IDS2017.zip
unzip datasets/CIC-IDS2017.zip -d datasets/CIC-IDS2017
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess Historical Data

```bash
cd scripts
python historical_data.py
```
Merges and preprocesses raw CSV files from the CIC-IDS2017 dataset into `historical_data.csv`.

### 2. Download Pretrained LLM

```bash
cd scripts
python check_model_download.py
```
Downloads **Google’s FLAN-T5 base model** for anomaly detection.

### 3. Reduce Dataset and Fine-Tune (Optional)

```bash
cd scripts
python fine_tune_reduced.py
```
Simultaneously **creates a smaller, stratified dataset** for rapid experimentation **and fine-tunes** the anomaly detection model on it. Generates `reduced_data.csv` and updates model weights accordingly.

### 4. Fine-Tune the Model

```bash
cd scripts
python fine_tune_model.py
```
Fine-tunes the anomaly detection model on the **full or reduced dataset**.

### 5. Run Inference

```bash
cd scripts
python inference.py
```
Specify your log file, or use the `RealTimeAnalyzer` class for **live network monitoring**.

---

## Project Structure

* `datasets/` – Raw, cleaned, and reduced datasets.
* `scripts/` – Python scripts for data preprocessing, model training, and inference.
* `docs/` – Documentation, diagrams, and project artifacts.

---

## Requirements

* **Python 3.8+**
* Key packages: `pandas`, `numpy`, `scikit-learn`, `transformers`, `peft`, `datasets`, `tqdm`

```bash
pip install -r requirements.txt
```

---

## References

* [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
* [Business Requirements Document (HP\_BRD.pdf)](https://github.com/rajkadakia/network-anomaly-detection/blob/main/docs/HP_BRD.pdf)
