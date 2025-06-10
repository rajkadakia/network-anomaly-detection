import os
import datetime

# WSL path to the syslog-ng log file
log_file_path = r"\\wsl$\Ubuntu\var\log\tcp-network-logs.log"

# Destination folder on Windows to store batch files
output_dir = r"D:\raj\hp internship\network logs"

def get_next_filename():
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("batch_") and f.endswith(".txt")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(output_dir, f"batch_{next_number:03}.txt")

def create_batch_file():
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        last_20 = lines[-20:] if len(lines) >= 20 else lines
        if not last_20:
            print("No logs to save.")
            return

        batch_path = get_next_filename()
        with open(batch_path, 'w') as out_file:
            out_file.writelines(last_20)

        print(f" Batch file created: {batch_path}")

    except FileNotFoundError:
        print(f" Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f" Unexpected error: {e}")

if __name__ == "__main__":
    create_batch_file()
