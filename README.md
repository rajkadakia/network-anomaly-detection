# Network Anomaly Detection

## Overview

This project implements a network anomaly detection pipeline using machine learning and large language models (LLMs). It is designed to process network logs, extract features, and detect various types of network attacks in real-time or batch mode. The system is built to be extensible and leverages the CIC-IDS2017 dataset for training and evaluation.

### Functional Architecture

The pipeline consists of the following stages (see Functional_Diagram.jpg):

1. **Log Collection**: Network logs are collected from devices such as firewalls, routers, web servers, and proxies.
2. **Log Parsing & Preprocessing**: Raw logs are parsed and preprocessed for feature extraction.
3. **Feature Extraction**: Relevant features are extracted from the logs for model input.
4. **Detection Engine**: A fine-tuned LLM (e.g., T5) analyzes the features and classifies network events as benign or various attack types.
5. **Explanation Module**: Generates human-readable alerts for detected anomalies.
6. **Model Training Pipeline**: Supports training and fine-tuning on labeled datasets.

## Dataset

- The main dataset used is [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), located in the datasets directory.
- Preprocessed and reduced versions are available for faster experimentation.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/rajkadakia/network-anomaly-detection.git
   cd network-anomaly-detection
   ```

2. **Set up a Python virtual environment**:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```

3. **Download the CIC-IDS2017 dataset**:
   - Using `wget`:
     ```sh
     wget https://www.unb.ca/cic/datasets/machine-learning-ids-2017.html -O datasets/CIC-IDS2017.zip
     unzip datasets/CIC-IDS2017.zip -d datasets/CIC-IDS2017
     ```
   - Or using `curl`:
     ```sh
     curl -L https://www.unb.ca/cic/datasets/machine-learning-ids-2017.html -o datasets/CIC-IDS2017.zip
     unzip datasets/CIC-IDS2017.zip -d datasets/CIC-IDS2017
     ```

4. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Historical Data

To preprocess and merge the raw CIC-IDS2017 CSV files:
```sh
cd scripts
python historical_data.py
```
This generates historical_data.csv.

### 2. Download Pretrained Model

To download the base model from HuggingFace:
```sh
cd scripts
python check_model_download.py
```
We use Google's FLAN T5 model as our base.

### 3. Reduce Dataset Size (Optional) and Fine-Tune the Model

To create a smaller, stratified dataset for quick experiments and then fine-tune the anomaly detection model:
```sh
cd scripts
python fine_tune_reduced.py
```
This also generates reduced_data.csv.

### 4. Fine-Tune the Model

To fine-tune the anomaly detection model:
```sh
cd scripts
python fine_tune_model.py
```
You can modify the script to use either the full or reduced dataset.


### 5. Run Inference

To analyze network logs for anomalies:
```sh
cd scripts
python inference.py
```
Edit the script to specify your log file or use the `RealTimeAnalyzer` class for real-time analysis.

## Project Structure

- datasets: Contains raw, cleaned, and reduced datasets.
- scripts: Python scripts for data processing, model training, and inference.
- docs: Project documentation and diagrams.

## Requirements

- Python 3.8+
- See requirements.txt for required packages (pandas, numpy, scikit-learn, transformers, peft, datasets, tqdm).

## References

- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- See the [business requirements document](https://github.com/rajkadakia/network-anomaly-detection/blob/main/docs/HP_BRD.pdf) for business requirements and detailed documentation.
