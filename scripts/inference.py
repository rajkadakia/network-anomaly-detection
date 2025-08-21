import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime

class RealTimeAnalyzer:
    def __init__(self, model_path="../models/network_anomaly_detector"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        
        self.label_map = {
            0: "BENIGN",
            1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk",
            5: "DoS Slowhttptest", 6: "DoS slowloris", 7: "FTP-Patator",
            8: "Heartbleed", 9: "Infiltration", 10: "PortScan",
            11: "SSH-Patator", 12: "Web Attack Brute Force",
            13: "Web Attack Sql Injection", 14: "Web Attack XSS"
        }
    
    def preprocess_log(self, log_line):
        """Convert raw syslog to model input format"""
        return f"Analyze network logs: {log_line.split(':', 3)[-1].strip()}"
    
    def analyze_batch(self, batch_file):
        """Process log batch with anomaly detection"""
        with open(batch_file) as f:
            logs = [self.preprocess_log(line) for line in f]
        
        inputs = self.tokenizer(
            logs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs)
        predictions = [self.label_map[int(self.tokenizer.decode(oid))] for oid in outputs]
        
        return self.generate_report(batch_file, logs, predictions)
    
    def generate_report(self, batch_file, logs, predictions):
        """BRD-compliant analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_file": batch_file,
            "total_logs": len(logs),
            "anomalies": sum(1 for p in predictions if p != "BENIGN"),
            "breakdown": {k: predictions.count(k) for k in set(predictions)},
            "critical_alerts": [
                f"{logs[i]} → {predictions[i]}" 
                for i, p in enumerate(predictions) 
                if p in ["DDoS", "Heartbleed", "Infiltration"]
            ]
        }
        return report

if __name__ == "__main__":
    import sys
    analyzer = RealTimeAnalyzer()
    
    # Read batch from stdin (syslog-ng program() destination)
    batch = [line.strip() for line in sys.stdin]
    
    # Temporary batch file processing
    with open("/tmp/current_batch.log", "w") as f:
        f.write("\n".join(batch))
    
    report = analyzer.analyze_batch("/tmp/current_batch.log")
    print(json.dumps(report, indent=2))
