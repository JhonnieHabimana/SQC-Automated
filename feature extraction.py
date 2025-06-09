# Filepath
base_filepath = '/Users/admin/Desktop/ML'  
acceptable_path = os.path.join(base_filepath, 'acceptable')
query_path = os.path.join(base_filepath, 'query')
plotOn = 0  # Set to 0 if you don't want to plot figures
saveOn = 0  # Set to 1 if you want to save plots

# Files
acceptable_files = [os.path.join(acceptable_path, f) for f in os.listdir(acceptable_path) if f.endswith('.csv')]
query_files = [os.path.join(query_path, f) for f in os.listdir(query_path) if f.endswith('.csv')]
filelist = [(file, 'acceptable') for file in acceptable_files] + [(file, 'query') for file in query_files]

# Extract Patient ID from filename
def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    match = re.match(r'(\d+)_', filename)
    return match.group(1) if match else None

# Feature matrix
num_gait_features = 26  # Updated to 26 to include power spectrum features
X = np.full((len(filelist), num_gait_features), np.nan)  
labels = []
patient_ids = [] 

for iFile, (file, label) in enumerate(filelist):
    data = pd.read_csv(file, sep=',') 
    data.columns = data.columns.str.strip()  

    patient_id = extract_patient_id(file)
    patient_ids.append(patient_id)

    time = data['Time'].values
    acc_x = data['Acc_X'].values
    acc_y = data['Acc_Y'].values
    acc_z = data['Acc_Z'].values
    gyr_x = data['Gyr_X'].values
    gyr_y = data['Gyr_Y'].values
    gyr_z = data['Gyr_Z'].values
    acc_magnitude = data['Acc_Magnitude'].values
    gyr_magnitude = data['Gyr_Magnitude'].values

    number_strides = data['Number_Strides'].values[0]
    duration = data['Duration_s'].values[0]
    avg_stride_speed = data['Avg_Stride_Speed_mps'].values[0]

    # Handle missing columns
    avg_stride_length = data['AverageStrideLength'].values[0] if 'AverageStrideLength' in data.columns else np.nan
    avg_stride_duration = data['AverageStrideDuration'].values[0] if 'AverageStrideDuration' in data.columns else np.nan
    avg_stride_height_change = data['AverageStrideHeightChange'].values[0] if 'AverageStrideHeightChange' in data.columns else np.nan

    labels.append(label)

    # 1. Number of peaks in acceleration magnitude
    pkAmps, _ = find_peaks(acc_magnitude, distance=0.7 * 100, height=0.5 * np.max(acc_magnitude))
    X[iFile, 0] = len(pkAmps)

    # 2. Peak frequency of acceleration magnitude (Welch method)
    f, Pxx = welch(acc_magnitude, fs=100, nperseg=256)
    X[iFile, 1] = f[np.argmax(Pxx)]

    # 3. Signal jaggedness (average absolute difference between consecutive values)
    X[iFile, 2] = np.mean(np.abs(np.diff(acc_magnitude)))

    # 4. Signal entropy
    X[iFile, 3] = entropy(np.abs(np.diff(acc_magnitude)))

    # 5. Determinism (linearity measure using linear regression)
    model = LinearRegression().fit(np.arange(len(acc_magnitude)).reshape(-1, 1), acc_magnitude)
    X[iFile, 4] = model.coef_[0]

    # 6. Cadence (steps per minute)
    X[iFile, 5] = number_strides / (duration / 60) if duration > 0 else np.nan  

    # 7. Average stride speed
    X[iFile, 6] = avg_stride_speed  

    # 8. Stride time variability
    X[iFile, 7] = np.std(np.diff(time))  

    # 9. Duration of walking bout
    X[iFile, 8] = duration  

    # 10. Number of strides
    X[iFile, 9] = number_strides  

    # 11. Average stride length
    X[iFile, 10] = avg_stride_length  

    # 12. Average stride duration
    X[iFile, 11] = avg_stride_duration  

    # 13. Average stride height change
    X[iFile, 12] = avg_stride_height_change  

    # 14. Variability in stride speed
    X[iFile, 13] = np.std(data['Avg_Stride_Speed_mps']) if 'Avg_Stride_Speed_mps' in data.columns else np.nan  

    # 15. Ratio of acceleration peaks to strides
    X[iFile, 14] = len(pkAmps) / number_strides if number_strides > 0 else 0  

    # 16. Ratio of stride duration to stride length (gait efficiency)
    X[iFile, 15] = avg_stride_duration / avg_stride_length if avg_stride_length > 0 else np.nan  

    # 17. Variability in acceleration magnitude (measures consistency)
    X[iFile, 16] = np.std(acc_magnitude)  

    # 18. Variability in gyroscope magnitude (measures steadiness)
    X[iFile, 17] = np.std(gyr_magnitude)  

    # 19. Stride symmetry (difference between left and right step times) compare 
    if 'Stride_Start' in data.columns:
    # Get all stride start times
        stride_starts = data['Stride_Start'].dropna().values

        # Calculate durations between consecutive strides
        stride_durations = np.diff(stride_starts)

        if len(stride_durations) >= 2:
         # Separate odd and even stride durations
            odd_strides = stride_durations[1::2]  # Every second stride (odd indices)
            even_strides = stride_durations[0::2]  # Every second stride (even indices)

            # Ensure they are the same length for comparison
            min_length = min(len(odd_strides), len(even_strides))
            odd_strides = odd_strides[:min_length]
            even_strides = even_strides[:min_length]

            # Calculate symmetry as the mean absolute difference between odd and even strides
            stride_symmetry = np.mean(np.abs(odd_strides - even_strides))

            X[iFile, 18] = stride_symmetry
        else:
            X[iFile, 18] = np.nan  # Not enough strides to calculate symmetry
    else:
     X[iFile, 18] = np.nan  # Column missing
  

    # 20. Ratio of turns to total strides (checks for turning behavior)
    num_turns = len(data.dropna(subset=['Turn_Start'])) if 'Turn_Start' in data.columns else 0
    X[iFile, 19] = num_turns / number_strides if number_strides > 0 else 0  

    # Power Spectrum Features

    # 21. Total Power (sum of power spectrum)
    X[iFile, 20] = np.sum(Pxx)

    # 22. Spectral Entropy (entropy of power distribution)
    X[iFile, 21] = entropy(Pxx)

    # 23. Spectral Edge Frequency (95% of power)
    cumulative_power = np.cumsum(Pxx) / np.sum(Pxx)
    X[iFile, 22] = f[np.searchsorted(cumulative_power, 0.95)]

    # 24. Dominant Frequency Power Ratio (peak power / total power)
    X[iFile, 23] = np.max(Pxx) / np.sum(Pxx)

    # 25. Bandwidth (range of significant frequencies)
    X[iFile, 24] = f[np.argmax(Pxx)] - f[np.argmax(Pxx > 0.01 * np.max(Pxx))]

    # 26. Gyroscope Peak Frequency
    f_gyr, Pxx_gyr = welch(gyr_magnitude, fs=100, nperseg=256)
    X[iFile, 25] = f_gyr[np.argmax(Pxx_gyr)]

columns = [
    'Patient_ID', 'Number_of_Peaks', 'Peak_Frequency', 'Signal_Jaggedness', 'Signal_Entropy', 'Determinism', 
    'Cadence', 'Avg_Stride_Speed', 'Stride_Time_Variability', 'Duration', 'Number_Strides',
    'Avg_Stride_Length', 'Avg_Stride_Duration', 'Avg_Stride_Height_Change', 'Stride_Speed_Variability', 
    'Peak_to_Stride_Ratio', 'Stride_Duration_to_Length', 'Acc_Magnitude_Variability', 
    'Gyr_Magnitude_Variability', 'Stride_Symmetry', 'Turn_to_Stride_Ratio',
    'Total_Power', 'Spectral_Entropy', 'Spectral_Edge_Frequency', 'Dominant_Frequency_Ratio',
    'Bandwidth', 'Gyroscope_Peak_Frequency', 'Label'
]

# Construct DataFrame
feature_df = pd.DataFrame(X, columns=columns[1:-1])
feature_df.insert(0, 'Patient_ID', patient_ids) 
feature_df['Label'] = labels

# Save to CSV
output_path = os.path.join(base_filepath, 'gait_features.csv')
feature_df.to_csv(output_path, index=False)
print(f"Feature extraction complete. Saved to {output_path}")
