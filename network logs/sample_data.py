import pandas as pd
import random

# Step 1: File path (adjust this to your path)
file_path = r'D:\raj\hp internship\historical_data.csv'


# Step 2: Sample size and chunk settings
sample_size = 2000
chunk_size = 100000  # Reads 100,000 rows at a time
sampled_rows = []

# Step 3: Loop through CSV in chunks
try:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_sample = chunk.sample(n=min(sample_size, len(chunk)), random_state=random.randint(1, 10000))
        sampled_rows.append(chunk_sample)
        if sum(len(df) for df in sampled_rows) >= sample_size:
            break

    # Step 4: Concatenate and keep only first 2000 rows
    final_sample = pd.concat(sampled_rows).head(sample_size)

    # Step 5: Save output
    final_sample.to_csv('sampled_output.csv', index=False)
    print("✅ Sampled 2000 records to 'sampled_output.csv'")

except Exception as e:
    print("❌ Error:", e)
