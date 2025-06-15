import pandas as pd

df = pd.read_csv("merged_dataset.csv")
df.columns = df.columns.str.strip()  # Remove whitespace from column names

print("Labels found:", df['Label'].unique())

# Show label distribution
print("\n Label Counts:")
label_counts = df['Label'].value_counts()
print(label_counts)

# Set max sample size
MAX_SAMPLES = 5000

# Create a list to hold dataframes
sampled_dfs = []

# Loop over each label
for label, count in label_counts.items():
    subset = df[df['Label'] == label]
    if count > MAX_SAMPLES:
        sampled = subset.sample(n=MAX_SAMPLES, random_state=42)
    else:
        sampled = subset  # keep all if less than 5000
    sampled_dfs.append(sampled)

# Concatenate all samples
final_sample = pd.concat(sampled_dfs, ignore_index=True)

# Save to CSV
final_sample.to_csv("sampled_max5000_per_label.csv", index=False)
print(f"\n Done! Saved to 'sampled_max5000_per_label.csv' with:")
print(final_sample['Label'].value_counts())
