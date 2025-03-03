import os
import h5py
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from Bio import SeqIO
from io import StringIO

# Load the trained DNN model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load signal data from FAST5 file
def load_fast5_signal(fast5_path):
    try:
        with h5py.File(fast5_path, 'r') as f:
            # Search for all possible Read_ids and retrieve the Signal data
            for read_id in f['/Raw/Reads']:
                signal_data = f[f'/Raw/Reads/{read_id}/Signal'][:]
                return signal_data
    except Exception as e:
        print(f"Error loading {fast5_path}: {e}")
        return None

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

# Extract features using a sliding window approach
def extract_features(signal_data, window_size):
    features = []
    for i in range(len(signal_data) - window_size + 1):  # Adjusted to slide within bounds
        window = signal_data[i:i + window_size]
        mean = np.mean(window)
        std_dev = np.std(window)
        max_val = np.max(window)
        min_val = np.min(window)
        # Ensure only 4 features are extracted (mean, std, max, min)
        features.append([mean, std_dev, max_val, min_val])
    return np.array(features)

# Predict polyA regions using the DNN model
def detect_polyA(model, signal_data, window_size, step_size):
    features = []
    polyA_regions = []
    
    # Sliding window approach with overlap
    for i in range(0, len(signal_data) - window_size + 1, step_size):
        window = signal_data[i:i + window_size]
        # Extract features for each window (4 features)
        window_features = extract_features(window, window_size)
        features.append(window_features)
    
    # Convert features to numpy array and ensure it has the correct shape (None, 4)
    features = np.array(features)
    features = features.squeeze(axis=1)  # Remove the second dimension if its size is 1

    # Get predictions for all windows
    predictions = (model.predict(features) > 0.5).astype("int32")
    
    # Identify polyA regions (where predictions are 1)
    for i, pred in enumerate(predictions):
        if pred == 1:
            start = i * step_size
            end = start + window_size
            polyA_regions.append([start, end])

    return polyA_regions

# Merge overlapping polyA regions
def merge_regions(polyA_regions):
    if not polyA_regions:
        return []
    
    # Sort by start position
    polyA_regions.sort(key=lambda x: x[0])
    
    merged_regions = [polyA_regions[0]]
    for current in polyA_regions[1:]:
        prev = merged_regions[-1]
        if current[0] <= prev[1]:  # Overlapping or touching
            prev[1] = max(prev[1], current[1])  # Merge by extending the end position
        else:
            merged_regions.append(current)
    
    return merged_regions

# Filter and merge polyA regions based on custom conditions
def filter_and_merge_regions(polyA_regions, min_pos=6000, max_pos=20000):
    if len(polyA_regions) == 1:
        return polyA_regions  # 只有一个区域，直接返回

    # 过滤出 6000 - 20000 之间的区域
    filtered_regions = [region for region in polyA_regions if min_pos <= region[0] and region[1] <= max_pos]

    if filtered_regions:
        # 如果有多个符合条件的区域，合并
        merged_regions = merge_regions(filtered_regions)
        return [[merged_regions[0][0], merged_regions[-1][1]]]

    # 如果没有符合 6000-20000 的区域，则需要特殊处理
    below_min = [region for region in polyA_regions if region[1] < min_pos]
    above_max = [region for region in polyA_regions if region[0] > max_pos]

    if below_min and not above_max:
        # 只有低于 6000 的区域，选取最接近 6000 的区域（即最大 `region[1]`）
        return [max(below_min, key=lambda x: x[1])]

    if above_max and not below_min:
        # 只有高于 20000 的区域，选取最远离 20000 的区域（即最大 `region[0]`）
        return [max(above_max, key=lambda x: x[0])]

    return []  # 理论上不会发生，但如果 `polyA_regions` 为空，则返回空列表


# Function to find polyA sequences in a single read sequence
def sliding_window(sequence, window_size=7, allowed_mismatch=2):
    result = []
    temp_seq = []
    
    i = 0
    while i <= len(sequence) - window_size:
        window = sequence[i:i + window_size]
        non_a_count = sum(1 for base in window if base != 'A')
        
        # 判断是否符合条件
        if non_a_count <= allowed_mismatch:
            if temp_seq:
                temp_seq.append(sequence[i + window_size - 1])  # 追加当前窗口最后一个碱基
            else:
                temp_seq = list(window)  # 记录第一个符合条件的窗口
        else:
            if temp_seq:
                result.append(''.join(temp_seq))  # 如果有临时序列，加入到结果
                temp_seq = []  # 清空临时序列
        i += 1
    
    # 如果结束时还有符合条件的窗口序列
    if temp_seq:
        result.append(''.join(temp_seq))
    
    return result

def is_polya_valid(polya_positions, sequence):
    # 判断polya_positions是否非空且sequence是否非空
    if polya_positions and sequence:
        return True
    else:
        return False

# Main function to detect polyA regions
def main(fast5_dir, model_path, result_output_file, window_count):
    # Load the trained model
    model = load_model(model_path)
    
    polyA_results = []
    
    # Iterate through all FAST5 files in the directory
    for fast5_file in os.listdir(fast5_dir):
        if fast5_file.endswith('.fast5'):
            fast5_path = os.path.join(fast5_dir, fast5_file)
            print(f"Processing {fast5_path}...")
            
            # Load the signal data from the FAST5 file
            signal_data = load_fast5_signal(fast5_path)
            if signal_data is None:
                continue  # Skip if signal data is not found
            
            # Load the fastq sequences from the FAST5 file
            fastq_sequences = extract_fastq(fast5_path)
           
            # Process FASTQ sequences for polyA detection
            all_results = []
            for fastq_data in fastq_sequences:
                # 使用 StringIO 将字符串包装成文件句柄
                fastq_handle = StringIO(fastq_data)
                # 使用 SeqIO.parse 解析 FastQ 数据
                for seq_record in SeqIO.parse(fastq_handle, "fastq"):
                    sequence = str(seq_record.seq)
                    results=sliding_window(sequence)
                    all_results.extend(results)
                    #print(all_results)
            if not all_results: 
                last_results=""
            else:
                last_results = all_results[-1]
            # Calculate window size and step size for signal-based polyA region detection
            window_size = len(signal_data) // window_count  # Integer division to get window size
            step_size = window_size // 2  # Step size is half of the window size
            
            # Detect polyA regions using the model
            polyA_regions = detect_polyA(model, signal_data, window_size, step_size)
            
            # Merge overlapping polyA regions
            merged_regions = merge_regions(polyA_regions)
            
            # Filter and merge polyA regions based on custom conditions
            filtered_and_merged_regions = filter_and_merge_regions(merged_regions)
            
            polya_is_valid = is_polya_valid(filtered_and_merged_regions, last_results)
            # Append both polyA sequences and polyA regions (split into tail_start and tail_end)
            if not polyA_regions:
                tail_start, tail_end = "",""
            else:
                for region in filtered_and_merged_regions:
                    tail_start, tail_end = region
                    #valid = tail_end - tail_start > 50  # Example validation condition
            polyA_results.append([fast5_file, tail_start, tail_end, last_results, polya_is_valid])
            
            # Append both polyA sequences and polyA regions to the results for this file
            #polyA_results.append([fast5_file, filtered_and_merged_regions, last_results,polya_is_valid])
    
    # Save all detected polyA regions and sequences to a CSV file
    save_results(polyA_results, result_output_file)

# Save the detected polyA sequences and regions to a CSV file
def save_results(polyA_results, output_file):
    # Create a DataFrame with file path, polyA regions, and polyA sequences
    df = pd.DataFrame(polyA_results, columns=['file_path', 'tail_start', 'tail_end','polyA_sequences',"polya_is_valid"])
    
    # Convert lists of polyA sequences to strings for CSV storage
    #df['polyA_sequences'] = df['polyA_sequences'].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Detect polyA regions in FAST5 files using the DNN model.")
    
    # Define the arguments for the FAST5 directory, model path, window count, and result output
    parser.add_argument('--fast5_dir', type=str, required=True, help='Directory containing FAST5 files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained DNN model')
    parser.add_argument('--result_output_file', type=str, required=True, help='Path to save the detection results')
    parser.add_argument('--window_count', type=int, required=True, help='Number of windows to divide the signal into')
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args.fast5_dir, args.model_path, args.result_output_file, args.window_count)
