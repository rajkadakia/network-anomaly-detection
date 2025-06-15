import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Step 1: Load the dataset
csv_path = 'logs_text_for_rag.csv'  # Adjust if needed
df = pd.read_csv(csv_path)

if 'text' not in df.columns:
    raise ValueError("❌ The 'text' column is missing in your CSV file.")

texts = df['text'].tolist()
print(f"✅ Loaded {len(texts)} log entries.")

# Step 2: Load the SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'  # You can change this to other SBERT models
print(f"🔍 Loading model: {model_name}")
model = SentenceTransformer(model_name)

# Step 3: Generate embeddings
print("⚙️ Generating embeddings... this might take a while ⏳")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

# Step 4: Save embeddings and labels (optional for next FAISS step)
np.save('log_embeddings.npy', embeddings)
df['label'].to_csv('log_labels.csv', index=False)

print("✅ Embeddings saved to 'log_embeddings.npy'")
print("✅ Labels saved to 'log_labels.csv'")