import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import pywt
import numpy as np
from scipy.signal import lfilter, correlate
from scipy.signal import butter, filtfilt

# Define folder path and output file name
folder_path = 'F:/dataset/ecgppgBP/archive (1)/part2/smaller_csv'
output_file = '8may23.csv'

# Define feature names
feature_names = ['ptt', 'pat', 'mean_ppg', 'std_ppg', 'mean_ecg', 'std_ecg', 'sbp', 'dbp', 'hr', 'map', 'sys_bp', 'dia_bp', 'systolic_peak_amp', 'width_one_pulse', 'mean_peak_time_diff', 'area_under_ppg', 'crest_time', 'inflection_time', 'mfl_feat_1', 'mfl_feat_2', 'shannon_entropy', 'hjorth_mobility', 'hjorth_complexity']

# Initialize list of feature vectors
features_list = []

# Loop through all CSV files in folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        try:
            # Load data from CSV file
            data = pd.read_csv(os.path.join(folder_path, file_name), header=None)

            # Extract PPG, ECG, and BP waveforms
            ppg = data.iloc[0].values
            ecg = data.iloc[2].values
            bp = data.iloc[1].values

            # Calculate PTT and PAT
            ppg_peaks, _ = find_peaks(ppg, height=1.5)
            ecg_peaks, _ = find_peaks(ecg, height=0.7)

            ppg_peak_time = ppg_peaks[1] / 125.0
            ecg_r_peak_time = ecg_peaks[1] / 125.0

            ptt = ppg_peak_time - ecg_r_peak_time
            pat = ppg_peak_time - ecg_r_peak_time

            # Calculate other features
            mean_ppg = np.mean(ppg)
            std_ppg = np.std(ppg)
            mean_ecg = np.mean(ecg)
            std_ecg = np.std(ecg)

            # Add known BP relationships as features
            sbp = 1.95 * ptt + 20.3
            dbp = 1.57 * pat - 12.8
            map = np.mean(bp)

            # Calculate systolic and diastolic BP
            sys_bp = np.max(bp)
            dia_bp = np.min(bp)

            # Calculate heart rate
            rr_interval = ecg_peaks[1] - ecg_peaks[0]
            hr = 60.0 / (rr_interval / 125.0)

            # Calculate additional PPG features
            ppg_peaks, _ = find_peaks(ppg, height=1.5)
            systolic_peak_amp = ppg[ppg_peaks[0]]

            peak_distances = np.diff(ppg_peaks)
            width_one_pulse = np.mean(peak_distances) / 125.0

            peak_times = ppg_peaks / 125.0
            peak_time_diffs = np.diff(peak_times)
            
            # Calculate additional PPG features (continued)
            mean_peak_time_diff = np.mean(np.diff(ppg_peaks)) / 125.0
            area_under_ppg = np.trapz(ppg)

            crest_time = peak_times[0]
            inflection_time = peak_times[1]
    
            # Calculate MFL features using discrete wavelet transform
            cA, cD = pywt.dwt(ppg, 'db4')
            mfl_feat_1 = np.sum(np.abs(cA))
            mfl_feat_2 = np.sum(np.abs(cD))
    
            # Calculate Shannon entropy, Hjorth mobility, and Hjorth complexity
            norm_ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
            shannon_entropy = -np.sum(norm_ppg * np.log2(norm_ppg + 1e-10))
    
            diff1_ppg = np.diff(ppg)
            diff2_ppg = np.diff(diff1_ppg)
    
            hjorth_mobility = np.sqrt(np.var(diff1_ppg) / np.var(ppg))
            hjorth_complexity = np.sqrt(np.var(diff2_ppg) / np.var(diff1_ppg))
    
            # Append feature vector to list
            feature_vector = [ptt, pat, mean_ppg, std_ppg, mean_ecg, std_ecg, sbp, dbp, hr, map, sys_bp, dia_bp, systolic_peak_amp, width_one_pulse, mean_peak_time_diff, area_under_ppg, crest_time, inflection_time, mfl_feat_1, mfl_feat_2, shannon_entropy, hjorth_mobility, hjorth_complexity]
    
            features_list.append(feature_vector)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Create pandas DataFrame from feature list and save to CSV file
df = pd.DataFrame(features_list, columns=feature_names)
df.to_csv(output_file, index=False) 