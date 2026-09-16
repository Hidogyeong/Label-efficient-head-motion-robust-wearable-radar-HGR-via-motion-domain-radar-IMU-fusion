##########################################
# FMCW Radar Data Processing
##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

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
    data = []
    for file in csv_files:
        loaded_data = np.loadtxt(file, delimiter=',')
        data.append(loaded_data)
    return np.array(data)

def process_radar_data(data, num_chirps=32, num_rx_antennas=3, num_samples=64, num_beams=27, max_angle_degrees=40):
    """
    Process radar data to generate range, doppler, and angle information.
    """
    # Reshape data to (num_frames, 3, 32, 64)
    data = data.reshape(-1, num_rx_antennas, num_chirps, num_samples)
    
    # Initialize Doppler and BeamForming algorithms
    doppler = DopplerAlgo(num_samples, num_chirps, num_rx_antennas)
    dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)

    range_data = []
    doppler_data = []
    horizontal_angles = []
    vertical_angles = []
    velocity_indices = []
    distance_indices = []

    # Iterate through each frame
    for frame in data:
        rd_spectrum_h = np.zeros((num_samples, 2 * num_chirps, num_rx_antennas), dtype=complex)
        rd_spectrum_v = np.zeros((num_samples, 2 * num_chirps, num_rx_antennas), dtype=complex)
        data_all_antennas = []

        # Process each antenna's data in the current frame
        for i_ant in range(num_rx_antennas):
            mat = frame[i_ant, :, :]
            dfft_dbfs_complex = doppler.compute_doppler_map(mat, i_ant)
            dfft_dbfs = np.abs(dfft_dbfs_complex)
            data_all_antennas.append(dfft_dbfs)

            if i_ant == 0:
                rd_spectrum_v[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 1:
                rd_spectrum_h[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 2:
                rd_spectrum_v[:, :, 1] = dfft_dbfs_complex
                rd_spectrum_h[:, :, 1] = dfft_dbfs_complex

        # Angle calculations
        rd_beam_formed_h = dbf.run(rd_spectrum_h)
        rd_beam_formed_v = dbf.run(rd_spectrum_v)

        beam_range_energy_h = np.linalg.norm(rd_beam_formed_h, axis=1) / np.sqrt(num_beams)
        beam_range_energy_v = np.linalg.norm(rd_beam_formed_v, axis=1) / np.sqrt(num_beams)

        # Calculate horizontal and vertical angles
        top_7_indices_h = np.argpartition(beam_range_energy_h.flatten(), -3)[-3:]
        top_7_angles_h = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[top_7_indices_h % num_beams]
        angle_degrees_h = np.mean(top_7_angles_h)

        top_7_indices_v = np.argpartition(beam_range_energy_v.flatten(), -3)[-3:]
        top_7_angles_v = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[top_7_indices_v % num_beams]
        angle_degrees_v = np.mean(top_7_angles_v)

        horizontal_angles.append(angle_degrees_h)
        vertical_angles.append(angle_degrees_v)

        # Range-Doppler
        if data_all_antennas:
            averaged_data = np.mean(data_all_antennas, axis=0)
            summed_velocity_all = np.sum(averaged_data, axis=0)
            velocity_index = np.argmax(summed_velocity_all)
            summed_distance_all = np.sum(averaged_data, axis=1)
            distance_index = np.argmax(summed_distance_all)
            velocity_indices.append(velocity_index - 32)
            distance_indices.append(distance_index)

        # Calculate Doppler information
        doppler_data.append(velocity_index - 32)

        # Calculate range information
        range_data.append(distance_index)

    return np.array(range_data), np.array(doppler_data), np.array(horizontal_angles), np.array(vertical_angles)

def save_processed_data(range_data, doppler_data, horizontal_angles, vertical_angles, output_file):
    """
    Save the processed range, doppler, and angle data to a single CSV file.
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # Create a DataFrame with range, doppler, horizontal_angle, and vertical_angle for each frame
    df = pd.DataFrame({
        'frame': np.arange(len(horizontal_angles)),
        'range': range_data,
        'doppler': doppler_data,
        'horizontal_angle': horizontal_angles,
        'vertical_angle': vertical_angles
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

# def plot_radar_data(range_data, doppler_data, horizontal_angles, vertical_angles):
#     """
#     Plot the processed range, doppler, and angle data.
#     """
#     plt.figure(figsize=(15, 10))

#     # Plot range data
#     plt.subplot(2, 2, 1)
#     plt.plot(range_data, label='Range')
#     plt.title('Range Data')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Range Value')
#     plt.legend()

#     # Plot doppler data
#     plt.subplot(2, 2, 2)
#     plt.plot(doppler_data, label='Doppler', color='orange')
#     plt.title('Doppler Data')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Doppler Value')
#     plt.legend()

#     # Plot horizontal angles
#     plt.subplot(2, 2, 3)
#     plt.plot(horizontal_angles, label='Horizontal Angle', color='green')
#     plt.title('Horizontal Angle')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Angle (Degrees)')
#     plt.legend()

#     # Plot vertical angles
#     plt.subplot(2, 2, 4)
#     plt.plot(vertical_angles, label='Vertical Angle', color='red')
#     plt.title('Vertical Angle')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Angle (Degrees)')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# ============================ 
# Main Function
# ============================ 
if __name__ == "__main__":

    
    # Specify the directory containing radar CSV files
    radar_data_directory = "E:/dokyeong/Radar_rawdata/dk/02LtoR/None/"
    output_directory = "E:/dokyeong/Radar_RDAData/dk/02LtoR/None/"
    
    # Load data from CSV files
    radar_data_files = glob.glob(os.path.join(radar_data_directory, '**', '*.csv'), recursive=True)
    
    for file in radar_data_files:
        # Load data from a single CSV file
        radar_data = np.loadtxt(file, delimiter=',')
        
        # Process the loaded data to obtain range, doppler, and angles
        range_data, doppler_data, horizontal_angles, vertical_angles = process_radar_data(radar_data)
        
        # Save the processed data to a single CSV file
        relative_path = os.path.relpath(file, radar_data_directory)
        output_file = os.path.join(output_directory, relative_path)
        save_processed_data(range_data, doppler_data, horizontal_angles, vertical_angles, output_file)
        
        # Plot the processed range, doppler, and angle data
        # plot_radar_data(range_data, doppler_data, horizontal_angles, vertical_angles)
