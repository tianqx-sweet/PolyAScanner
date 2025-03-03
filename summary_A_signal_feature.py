import h5py
import numpy as np
import pandas as pd
import sys
import os

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std if std != 0 else signal

def extract_signal_features(fast5_path, tail_start, tail_end, segment_size=500):
    with h5py.File(fast5_path, 'r') as f:
        # 读取 raw signal，路径可能因数据格式不同有所变动
        raw_signal = f['Raw']['Reads'][list(f['Raw']['Reads'].keys())[0]]['Signal'][:]
        
        # Z标准化
        norm_signal = z_score_normalize(raw_signal)
        
        # 过滤 Z 标准化后在 [-3,3] 之间的值
        norm_signal = norm_signal[(norm_signal >= -3) & (norm_signal <= 3)]
        
        # 提取 tail_start 到 tail_end 的信号
        tail_signal = norm_signal[tail_start:tail_end]
        
        # 分割区间并计算统计量
        segments = [tail_signal[i:i+segment_size] for i in range(0, len(tail_signal), segment_size)]
        stats = [[round(np.max(seg), 4), round(np.min(seg), 4), round(np.mean(seg), 4), round(np.std(seg), 4)] for seg in segments]
        
        return stats

def process_csv(input_csv, fast5_dir):
    df = pd.read_csv(input_csv)
    results = []
    
    for _, row in df.iterrows():
        fast5_filename = os.path.basename(row['file_path'])
        fast5_fullpath = os.path.join(fast5_dir, fast5_filename)
        
        if not os.path.exists(fast5_fullpath):
            print(f"Warning: {fast5_fullpath} not found.")
            results.append(None)
            continue
        
        tail_start = int(row['tail_start'])
        tail_end = int(row['tail_end'])
        
        stats = extract_signal_features(fast5_fullpath, tail_start, tail_end)
        results.append(stats)
    
    df['signal_stats'] = results
    output_csv = input_csv.replace('.csv', '_processed.csv')
    df.to_csv(output_csv, index=False)
    print(f"Processed file saved as: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <fast5_directory>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    fast5_dir = sys.argv[2]
    process_csv(input_csv, fast5_dir)