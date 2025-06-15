import pandas as pd
import json

# === Path Setup ===
file_path = r'C:\Users\SONY\Pulakita\network-anomaly-detection\datasets\sampled_max5000_per_label.csv'
output_text_path = r'C:\Users\SONY\Pulakita\network-anomaly-detection\datasets\logs_text_for_rag.csv'
label_map_path = r'C:\Users\SONY\Pulakita\network-anomaly-detection\datasets\label_map.json'

# === Load and clean CSV ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from all column names

# === Extract unique labels and create label map (optional)
unique_labels = sorted(df['Label'].unique())
label_map = {i: label for i, label in enumerate(unique_labels)}

# === Save Label Map
with open(label_map_path, 'w') as f:
    json.dump(label_map, f)

# === Convert Each Row to RAG-Compatible Text ===
def row_to_text(row):
    features = [f"{col} = {row[col]}" for col in df.columns if col.lower() != 'label']
    label_col = 'Label' if 'Label' in row else 'label'
    label_text = row[label_col]  # Already string, no int()
    return f"Flow with {'; '.join(features)}. Label: {label_text}."

df['text'] = df.apply(row_to_text, axis=1)

# === Save Output ===
df[['text', 'Label' if 'Label' in df.columns else 'label']].to_csv(output_text_path, index=False)

# === Done ===
print(f"✅ Structured log text saved to: {output_text_path}")
print(f"✅ Label map saved to: {label_map_path}")
