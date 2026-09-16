##########################################
# FMCW Radar Data Processing
##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import spectrogram

from helpers.DopplerAlgo import *
from helpers.DigitalBeamForming import *

# ============================ 
# Load and Process Radar Data
# ============================ 

def load_csv_files(directory):
    """
    Load all CSV files from the specified directory.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
    data = []
    for file in csv_files:
        try:
            loaded_data = np.loadtxt(file, delimiter=',')
            data.append(loaded_data)
            print(f"Successfully loaded file: {file}")
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return np.array(data)

def process_radar_data(data, num_chirps=32, num_rx_antennas=3, num_samples=64, sample_rate_Hz=2e6):
    """
    Process radar data to generate spectrogram results for each antenna.
    """
    # Reshape data to (num_frames, 3, 32, 64)
    try:
        # print(f"Data shape before reshaping: {data.shape}")
        data = data.reshape(-1, num_rx_antennas, num_chirps, num_samples)
        # print(f"Data shape after reshaping: {data.shape}")
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        return {}
    
    # Split the data by antenna and reshape
    antenna_data = [data[:, i, :, :].reshape(-1, num_chirps * num_samples) for i in range(num_rx_antennas)]  # Flatten (32, 64) to (32*64)
    data = np.stack(antenna_data, axis=0)  # Reshape to (3, 40, 32 * 64)
    # print(f"Data shape after splitting, flattening, and stacking: {data.shape}")

    # Initialize rawData as a list of numpy arrays to store the IF signals for each antenna
    rawData = [np.zeros((data.shape[1], num_chirps * num_samples - num_samples - 1)) for _ in range(num_rx_antennas)]

    # Loop through each frame and antenna to calculate IF signals
    for frame_idx in range(data.shape[1]):  # 40 frames
        for antenna_idx in range(num_rx_antennas):
            current_frame = data[antenna_idx, frame_idx, :]
            rawData[antenna_idx][frame_idx] = [current_frame[rd + num_samples] - current_frame[rd] for rd in range(len(current_frame) - num_samples - 1)]

    # Flatten rawData for each antenna
    rawData = [antenna_data.flatten() for antenna_data in rawData]

    # Perform spectrogram on IF signals for each antenna
    spectrogram_results = {}
    for antenna_idx in range(1):
        try:
            f, t, Sxx = spectrogram(rawData[antenna_idx], fs=2000000, nfft=256)
            spectrogram_results[antenna_idx] = (f, t, Sxx)
        except Exception as e:
            print(f"Error processing spectrogram for antenna {antenna_idx}: {e}")

    return spectrogram_results

# ============================ 
# Main Function
# ============================ 
if __name__ == "__main__":
    
    # Specify the directory containing radar CSV files
    radar_data_directory = "E:/dokyeong/Radar_rawdata/"
    output_directory = "E:/dokyeong/Radar_STFT/"
    
    # Load data from CSV files
    radar_data_files = glob.glob(os.path.join(radar_data_directory, '**', '*.csv'), recursive=True)
    if not radar_data_files:
        print(f"No radar data files found in directory: {radar_data_directory}")
    
    for file in radar_data_files:
        try:
            # Load data from a single CSV file
            radar_data = np.loadtxt(file, delimiter=',')
            print(f"Processing file: {file}")
            
            # Process the loaded data to obtain spectrogram results for each antenna
            spectrogram_results = process_radar_data(radar_data)
            
            # Save spectrogram images with amplitude for each antenna
            base_filename = os.path.splitext(os.path.basename(file))[0]
            relative_path = os.path.relpath(file, radar_data_directory)
            output_dir = os.path.join(output_directory, os.path.dirname(relative_path))
            os.makedirs(output_dir, exist_ok=True)
            for antenna_idx, (f, t, Sxx) in spectrogram_results.items():
                # Save the image without displaying it
                plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='jet')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'{base_filename}_{antenna_idx}.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"Saved spectrogram image for antenna {antenna_idx} at {output_dir}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
