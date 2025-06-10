import socket
import time
import random

# Generate the list of trusted IPs
def generate_trusted_ips():
    base_ip = "192.168.1."
    return [f"{base_ip}{i}" for i in range(1, 101)]

# Generate a log line with a given fake source IP
def generate_log_line(src_ip):
    fields = [
        random.randint(1000000, 120000000),
        random.randint(1, 50),
        random.randint(1, 50),
        random.randint(50, 1000),
        random.randint(50, 1000),
        random.randint(0, 600),
        0,
        round(random.uniform(10, 200), 8),
        round(random.uniform(0, 200), 8),
        random.randint(100, 3000),
        0,
        round(random.uniform(10, 200), 8),
        round(random.uniform(0, 200), 8),
        round(random.uniform(10, 200), 8),
        round(random.uniform(0, 10), 8),
        random.randint(100000, 6000000),
        round(random.uniform(0, 6000000), 8),
        random.randint(1000000, 6000000),
        3,
        random.randint(1000000, 6000000),
        round(random.uniform(100000, 1000000), 8),
        round(random.uniform(100000, 2000000), 8),
        random.randint(1000000, 6000000),
        3,
        random.randint(90000, 1000000),
        round(random.uniform(10000, 30000), 8),
        round(random.uniform(10000, 50000), 8),
        random.randint(30000, 65000),
        589,
        0, 0, 0, 0,
        160, 92,
        round(random.uniform(0, 2), 8),
        round(random.uniform(0, 2), 8),
        0,
        517,
        round(random.uniform(50, 200), 8),
        round(random.uniform(10000, 25000), 8),
        round(random.uniform(10000, 25000), 8),
        0, 0, 0, 1, 0, 0, 0, 0, 0,
        71,
        round(random.uniform(0, 200), 8),
        round(random.uniform(10, 50), 8),
        round(random.uniform(10, 50), 8),
        41,
        160,
        0, 0, 0,
        0, 0, 0,
        7, 617, 4, 164,
        29200, 30, 6, 20,
        104815, 0, 104815, 104815,
        5394424, 0, 5394424, 5394424, 0,
        0  # Label
    ]
    return f"{src_ip}\t" + "\t".join(map(str, fields))

# Send logs every 4 seconds for 1 minute
def send_logs(host='127.0.0.1', port=601, duration_seconds=60, interval_seconds=4):
    trusted_ips = generate_trusted_ips()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    start_time = time.time()
    try:
        while True:
            current_time = time.time()
            if current_time - start_time >= duration_seconds:
                print("Finished sending logs for 1 minute.")
                break
            src_ip = random.choice(trusted_ips)
            log_line = generate_log_line(src_ip)
            sock.sendall(log_line.encode() + b"\n")
            print(f"Sent log from {src_ip}: {log_line[:100]}...")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        sock.close()

if __name__ == "__main__":
    send_logs()
