import pandas as pd
import numpy as np
import os
import joblib
import h5py
import io
import argparse
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Bio import SeqIO


# Load input data from CSV file
def load_input_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 函数：检查序列中是否包含6个连续的T或A
def contains_consecutive_T_or_A(sequence, threshold=6):
    # 检查是否包含至少5个连续的T或A
    return 'T' * threshold in sequence or 'A' * threshold in sequence

def extract_fastq_from_fast5(fast5_file):
    with h5py.File(fast5_file, 'r') as f:
        # 尝试多个可能的路径
        possible_paths = [
            '/Analyses/Basecall_1D_000/BaseCalled_template/Fastq',
            '/Analyses/Basecall_1D_000/Fastq',
            '/Analyses/Basecall_1D_000/BaseCalled_2D/Fastq'
        ]
        
        for path in possible_paths:
            if path in f:
                fastq_data = f[path][()]
                fastq_sequence = fastq_data.decode('utf-8')  # 转换为字符串

                # 使用 StringIO 将字符串转换为文件对象
                fastq_file_like = io.StringIO(fastq_sequence)

                # 使用 Biopython 解析 FastQ 格式数据
                for record in SeqIO.parse(fastq_file_like, "fastq"):
                    sequence = str(record.seq)
                    
                    # 检查序列中是否包含6个以上的连续T或A
                    if contains_consecutive_T_or_A(sequence):
                        return True  # 一旦找到，返回True
        return False  # 如果没有找到，返回False


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

# Extract features using a sliding window approach
def extract_features(signal_data, window_size=500): 
    features = []
    for i in range(len(signal_data) - window_size):
        window = signal_data[i:i + window_size]
        mean = np.mean(window)
        std_dev = np.std(window)
        max_val = np.max(window)
        min_val = np.min(window)
        features.append([mean, std_dev, max_val, min_val])
    return np.array(features)

# Build the deep neural network model
def build_dnn_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))  # First hidden layer
    model.add(Dense(64, activation='relu'))  # Second hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the neural network model
def train_model(data, labels, input_shape, batch_size=32, epochs=10):
    model = build_dnn_model(input_shape)
    model.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=1)
    return model

# Predict and evaluate the model
def predict_and_evaluate(model, test_data, true_labels):
    predictions = (model.predict(test_data) > 0.5).astype("int32")  # Convert probabilities to binary predictions
    #print(classification_report(true_labels, predictions))
    return predictions

# Save the results to a CSV file
def save_results(predictions, file_paths, output_file):
    with open(output_file, 'w') as f:
        f.write('file_path,tail_start,tail_end\n')
        for i, pred in enumerate(predictions):
            file_path = file_paths[i]
            tail_start, tail_end = pred
            f.write(f'{file_path},{tail_start},{tail_end}\n')

# Main function to execute the entire process
def main(input_file, fast5_dir, model_output_file, result_output_file, batch_size, epochs):
    # 1. Read the input CSV file
    df = load_input_data(input_file)
    
    # 2. Extract features and labels from FAST5 files
    features = []
    labels = []
    file_paths = []
    
    for index, row in df.iterrows():
        file_path = row['file_path']
        tail_start, tail_end = row['tail_start'], row['tail_end']
        
        # Concatenate the complete path to the FAST5 file
        fast5_path = os.path.join(fast5_dir, file_path.split('/')[-1])
        
        # Load the signal data from the FAST5 file
        signal_data = load_fast5_signal(fast5_path)
        if signal_data is None:
            continue  # Skip if signal data is not found
        
        # Extract features from the signal data
        signal_features = extract_features(signal_data, window_size=500)  # Window size updated to 500
        features.append(signal_features)
        
        # Create labels: 1 for A base region, 0 for non-A base region
        label = np.zeros(len(signal_features))  # Initialize with 0
        label[tail_start:tail_end] = 1  # Mark the A base region with 1
        labels.append(label)
        
        # Record the file path
        file_paths.append(file_path)
    
    # 3. Flatten the features and labels to prepare for training
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    # 5. Train the model using the neural network
    model = train_model(X_train, y_train, X_train.shape[1], batch_size=batch_size, epochs=epochs)

    # 6. Save the trained model
    model.save(model_output_file)
    
    # 7. Evaluate the model
    predictions = predict_and_evaluate(model, X_test, y_test)

    # 8. Save the prediction results
    save_results(predictions, file_paths, result_output_file)


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Train and predict on FAST5 signal data using a deep neural network.")
    
    # Define the arguments for the input file, fast5 directory, model output, and result output
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--fast5_dir', type=str, required=True, help='Directory containing FAST5 files')
    parser.add_argument('--model_output_file', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--result_output_file', type=str, required=True, help='Path to save the prediction results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')

    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args.input_file, args.fast5_dir, args.model_output_file, args.result_output_file, args.batch_size, args.epochs)
