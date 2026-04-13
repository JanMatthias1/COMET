import os
import re

# The log files we want to compare
log_files = {
    "Baseline COMET": "baseline_output.log",
    "GIN (Baseline)": "gin_output.log",
    "SchNet 3D": "schnet_output.log",
    "3D Infomax": "infomax_output.log"
}

def parse_log(log_path):
    stats = {
        "Epochs Run": 0,
        "Total Time (min)": "N/A",
        "Time/Epoch (sec)": "N/A",
        "Avg UPS (Updates/sec)": "N/A"
    }
    
    if not os.path.exists(log_path):
        return stats

    epochs = []
    ups_values = []
    total_time_sec = 0.0

    with open(log_path, 'r') as f:
        for line in f:
            # 1. Extract Epochs
            epoch_match = re.search(r'end of epoch (\d+)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            
            # 2. Extract UPS (Updates Per Second)
            ups_match = re.search(r"'ups': ([\d.]+)", line) or re.search(r"ups=([\d.]+)", line)
            if ups_match:
                ups_values.append(float(ups_match.group(1)))
                
            # 3. Extract Total Training Time
            time_match = re.search(r'done training in ([\d.]+) seconds', line)
            if time_match:
                total_time_sec = float(time_match.group(1))

    # Calculate final stats
    if epochs:
        stats["Epochs Run"] = max(epochs)
        
    if total_time_sec > 0:
        stats["Total Time (min)"] = round(total_time_sec / 60, 2)
        if stats["Epochs Run"] > 0:
            stats["Time/Epoch (sec)"] = round(total_time_sec / stats["Epochs Run"], 2)
            
    if ups_values:
        stats["Avg UPS (Updates/sec)"] = round(sum(ups_values) / len(ups_values), 2)
        
    return stats

# --- Generate the Table ---
print("\n" + "="*80)
print(f"{'Model Architecture':<25} | {'Epochs Run':<12} | {'Total Time (min)':<18} | {'Time/Epoch (sec)':<15}")
print("-" * 80)

for model_name, file_name in log_files.items():
    if os.path.exists(file_name):
        s = parse_log(file_name)
        print(f"{model_name:<25} | {str(s['Epochs Run']):<12} | {str(s['Total Time (min)']):<18} | {str(s['Time/Epoch (sec)']):<15}")
    else:
        print(f"{model_name:<25} | {'[Log file not found]':<52}")

print("="*80 + "\n")
