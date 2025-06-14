# LLM-Powered Network Anomaly Detection System
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)
![Contributors](https://img.shields.io/github/contributors/rajkadakia/network-anomaly-detection)

![Project Banner](docs/Functional_Diagram.jpg)

A machine learning system for real-time network anomaly detection using fine-tuned FLAN-T5 model, aligned with BRD requirements for enterprise network monitoring.

## 📋 Project Overview
This system implements the specifications from the [Business Requirements Document (BRD)](docs/HP_BRD.pdf) to:
- Automatically detect unusual patterns in network logs
- Process 2.8M+ records with 30%+ productivity improvement for IT teams
- Provide real-time analysis and visualizations of anomalies
- Support proactive network management through AI-driven insights

## 🏗️ System Architecture

### Components Overview
- **Data Ingestion**: syslog-ng server for real-time log collection
- **Preprocessing**: Feature extraction and normalization pipeline
- **ML Pipeline**: FLAN-T5 with LoRA fine-tuning
- **Inference Engine**: Real-time anomaly detection and classification
- **Monitoring**: Performance metrics and alerting system

### Supported Attack Types

| Attack Type | Description |
|-------------|-------------|
| BENIGN | Normal network traffic with no malicious intent |
| Bot | Automated malicious software performing unauthorized activities |
| DDoS | Distributed Denial of Service attacks overwhelming network resources |
| DoS GoldenEye | Layer 7 DoS attack targeting HTTP keep-alive connections |
| DoS Hulk | HTTP flooding attack generating massive web traffic volumes |
| DoS Slowhttptest | Slow HTTP attack exploiting partial HTTP requests |
| DoS slowloris | Connection-based DoS attack keeping connections open |
| FTP-Patator | Brute force attacks targeting FTP service credentials |
| Heartbleed | SSL/TLS vulnerability exploitation for memory disclosure |
| Infiltration | Advanced persistent threats with stealthy network penetration |
| PortScan | Network reconnaissance scanning for open ports and services |
| SSH-Patator | Brute force attacks against SSH authentication systems |
| Web Attack - Brute Force | Credential stuffing attacks on web applications |
| Web Attack - Sql Injection | Database manipulation through malicious SQL queries |
| Web Attack - XSS | Cross-site scripting attacks targeting web browsers |

## 📁 Repository Structure
```
network-anomaly-detection/
├── datasets/ # Training and sample data
├── docs/ # Documentation and diagrams
├── models/ # Pre-trained and fine-tuned models
├── network logs/ # Log generation and processing utilities
├── scripts/ # Main training and inference scripts
├── tests/ # Unit and integration tests
├── configs/ # Configuration files
└── README.md # This file
```

## 🛠️ Setup Instructions

### Clone Repository
```
git clone https://github.com/rajkadakia/network-anomaly-detection.git
cd network-anomaly-detection
```

### Handle Large Files
**Due to Git LFS limitations:**
```
# Download dataset manually
wget https://www.unb.ca/cic/datasets/ids-2017.html -O CIC-IDS2017.zip

# Create dataset directory
mkdir -p CIC-IDS2017
unzip CIC-IDS2017.zip -d CIC-IDS2017/
```

### Set up Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Prepare Training Dataset
```
# Generate processed dataset
python3 scripts/historical_data.py
```

## 🔧 Model Setup

### Download Base Model
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

# Download and cache locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save for offline use
model.save_pretrained("/models/flan-t5-base")
tokenizer.save_pretrained("/models/flan-t5-base")
```

## 🚀 Training Pipeline
### Full Dataset Training
```
python3 scripts/fine_tune_model.py
```

### Reduced Dataset Training
```
python3 scripts/fine_tune_reduced.py
```

## 🖥️ Usage
```
# Start real-time inference engine
python3 scripts/inference.py
```

## 📚 References
- [BRD Documentation](docs/HP_BRD.pdf)
- [Project Architecture](docs/Functional_Diagram.jpg)
- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
