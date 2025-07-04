import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder

def process_cicids2017(input_dir='../datasets/CIC-IDS2017', output_file='../datasets/historical_data.csv'):
    """Process and merge CIC-IDS2017 files with full preprocessing pipeline"""
    
    # 1. File discovery and validation
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # 2. Custom sorting function for chronological order
    def sorting_key(file_path):
        filename = os.path.basename(file_path).lower()
        day_order = {
            'monday': 1, 'tuesday': 2, 'wednesday': 3,
            'thursday': 4, 'friday': 5
        }
        parts = filename.split('-')
        day = day_order.get(parts[0], 99)
        time = 1 if 'morning' in filename else 2 if 'afternoon' in filename else 0
        sub_order = 1 if 'ddos' in filename else 2 if 'portscan' in filename else 0
        return (day, time, sub_order)

    # 3. Sort files chronologically
    sorted_files = sorted(files, key=sorting_key)

    # 4. Read files with header handling
    dfs = []
    column_names = None
    
    for idx, file in enumerate(sorted_files):
        print(f"Processing {os.path.basename(file)}")
        if idx == 0:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip().str.lower()
            column_names = df.columns.tolist()
        else:
            df = pd.read_csv(file, skiprows=1, header=None)
            df.columns = column_names
        
        dfs.append(df)

    # 5. Concatenate datasets
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # 6. Network-specific preprocessing
    # Drop non-essential network identifiers
    cols_to_drop = [
        'flow id', 'source ip', 'source port',
        'destination ip', 'destination port', 'timestamp'
    ]
    for col in cols_to_drop:
        if col in full_df.columns:
            full_df = full_df.drop(columns=col)

    # Handle infinite values in network metrics
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df = full_df.dropna()

    # 7. Label encoding for attack types
    le = LabelEncoder()
    full_df['label'] = le.fit_transform(full_df['label'])
    
    # 8. Final validation
    print("\nFinal dataset summary:")
    print(f"Total records: {len(full_df)}")
    print("Class distribution:")
    print(full_df['label'].value_counts())
    print(list(le.classes_))

# Final dataset summary:
# Total records: 2827876
# Class distribution:
# label
# 0     2271320
# 4      230124
# 10     158804
# 2      128025
# 3       10293
# 7        7935
# 11       5897
# 6        5796
# 5        5499
# 1        1956
# 12       1507
# 14        652
# 9          36
# 13         21
# 8          11
# Name: count, dtype: int64
# ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS']

    # 9. Save cleaned data
    full_df.to_csv(output_file, index=False)
    print(f"\nSaved preprocessed data to {output_file}")

if __name__ == "__main__":
    process_cicids2017()
