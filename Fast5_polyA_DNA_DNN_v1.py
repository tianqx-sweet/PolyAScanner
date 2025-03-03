import os
import h5py
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from Bio import SeqIO
from io import StringIO

# Load basecalled sequences from FAST5 file (BaseCalled_template)
def extract_fastq(fast5_file):
    fastq_sequences = []  # 存储提取到的FastQ序列
    try:
        with h5py.File(fast5_file, 'r') as f:
            # 定位到FastQ数据集
            #fastq_path = '/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'
            fastq_path = '/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'
            print(fastq_path)
            if fastq_path in f:
                fastq_data = f[fastq_path][()]  # 获取FastQ数据
                fastq_sequence = fastq_data.decode('utf-8')  # 转换为字符串
                fastq_sequences.append(fastq_sequence)
            else:
                print(f"FastQ path not found in the file: {fastq_path}")
    except Exception as e:
        print(f"Error reading Fast5 file: {e}")
    return fastq_sequences

# Load the trained DNN model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load signal data from FAST5 file
def load_fast5_signal(fast5_path):
    try:
        with h5py.File(fast5_path, 'r') as f:
            for read_id in f['/Raw/Reads']:
                return f[f'/Raw/Reads/{read_id}/Signal'][:]
    except Exception as e:
        print(f"Error loading {fast5_path}: {e}")
        return None

# Extract features using a sliding window approach
def extract_features(signal_data, window_size):
    return np.array([
        [np.mean(window), np.std(window), np.max(window), np.min(window)]
        for i in range(len(signal_data) - window_size + 1)
        for window in [signal_data[i:i + window_size]]
    ])

# Detect polyA/polyT regions
def detect_poly_regions(model, signal_data, window_size, step_size):
    features = np.array([
        extract_features(signal_data[i:i + window_size], window_size)
        for i in range(0, len(signal_data) - window_size + 1, step_size)
    ]).squeeze(axis=1)

    predictions = (model.predict(features) > 0.5).astype("int32")
    return [[i * step_size, i * step_size + window_size] for i, pred in enumerate(predictions) if pred == 1]

# Merge overlapping polyA/polyT regions
def merge_regions(regions):
    if not regions:
        return []
    regions.sort(key=lambda x: x[0])
    merged = [regions[0]]
    for curr in regions[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            prev[1] = max(prev[1], curr[1])
        else:
            merged.append(curr)
    return merged

# Filter and merge polyA regions
def filter_and_merge_regions_polyA(regions, min_pos=6000, max_pos=20000):
    filtered = [r for r in regions if min_pos <= r[0] <= max_pos and min_pos <= r[1] <= max_pos]
    return merge_regions(filtered) if filtered else []

# Filter and merge polyT regions
def filter_and_merge_regions_polyT(regions, min_pos=12000, max_pos=20000):
    filtered = [r for r in regions if min_pos <= r[0] <= max_pos and min_pos <= r[1] <= max_pos]
    return merge_regions(filtered) if filtered else []

# Sliding window search for polyA/polyT in sequences
def sliding_window(sequence, base, window_size=7, allowed_mismatch=2):
    result, temp_seq = [], []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        non_match = sum(1 for b in window if b != base)
        if non_match <= allowed_mismatch:
            temp_seq.append(window[-1]) if temp_seq else temp_seq.extend(window)
        elif temp_seq:
            result.append(''.join(temp_seq))
            temp_seq = []
    if temp_seq:
        result.append(''.join(temp_seq))
    return result

# Validate polyA/polyT presence
def is_valid(poly_positions, sequence):
    return bool(poly_positions and sequence)

# Main function
def main(fast5_dir, polyA_model_path, polyT_model_path, result_output_file, window_count):
    polyA_model, polyT_model = load_model(polyA_model_path), load_model(polyT_model_path)
    results = []
    
    for fast5_file in filter(lambda f: f.endswith('.fast5'), os.listdir(fast5_dir)):
        fast5_path = os.path.join(fast5_dir, fast5_file)
        signal_data = load_fast5_signal(fast5_path)
        
        fastq_sequences = [str(rec.seq) for seq_data in extract_fastq(fast5_path) for rec in SeqIO.parse(StringIO(seq_data), "fastq")]
        
        window_size, step_size = len(signal_data) // window_count, len(signal_data) // (2 * window_count)
        polyA_regions = filter_and_merge_regions_polyA(merge_regions(detect_poly_regions(polyA_model, signal_data, window_size, step_size)))
        polyT_regions = filter_and_merge_regions_polyT(merge_regions(detect_poly_regions(polyT_model, signal_data, window_size, step_size)))
        
        polyA_seqs, polyT_seqs = [], []
        for seq in fastq_sequences:
            polyA_seqs.extend(sliding_window(seq, 'A'))
            polyT_seqs.extend(sliding_window(seq, 'T'))
        
        last_polyA_seq = polyA_seqs[-1] if polyA_seqs else ""
        first_polyT_seq = polyT_seqs[0] if polyT_seqs else ""
        
        if last_polyA_seq:
            results.append([fast5_file, 'polyA', polyA_regions[0][0] if polyA_regions else "", polyA_regions[0][1] if polyA_regions else "", last_polyA_seq, True])
        elif first_polyT_seq:
            results.append([fast5_file, 'polyT', polyT_regions[0][0] if polyT_regions else "", polyT_regions[0][1] if polyT_regions else "", first_polyT_seq, True])
        elif polyA_regions:
            results.append([fast5_file, 'polyA', polyA_regions[0][0], polyA_regions[0][1], "", False])
        elif polyT_regions:
            results.append([fast5_file, 'polyT', polyT_regions[0][0], polyT_regions[0][1], "", False])
        else:
            results.append([fast5_file, "", "", "", "", False])
    
    save_results(results, result_output_file)

# Save results
def save_results(results, output_file):
    pd.DataFrame(results, columns=['file_path', 'is_polyA_or_polyT', 'polyA/T_start', 'polyA/T_end', 'polyA/T_sequences', 'valid_polyA/T']).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect polyA and polyT regions in FAST5 files using DNN models.")
    parser.add_argument('--fast5_dir', type=str, required=True, help='Directory containing FAST5 files')
    parser.add_argument('--polyA_model_path', type=str, required=True, help='Path to the trained polyA DNN model')
    parser.add_argument('--polyT_model_path', type=str, required=True, help='Path to the trained polyT DNN model')
    parser.add_argument('--result_output_file', type=str, required=True, help='Path to save the detection results')
    parser.add_argument('--window_count', type=int, required=True, help='Number of windows to divide the signal into')
    
    args = parser.parse_args()
    main(args.fast5_dir, args.polyA_model_path, args.polyT_model_path, args.result_output_file, args.window_count)
