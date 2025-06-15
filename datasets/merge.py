import pandas as pd
import os

# Folder containing your CSV files
folder = "sheets"  

# Read and merge all CSV files
all_dataframes = [pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.csv')]
merged = pd.concat(all_dataframes, ignore_index=True)

# Save the merged data to a single CSV file
merged.to_csv("merged_dataset.csv", index=False)

print("All CSV files merged and saved to 'merged_dataset.csv'")
