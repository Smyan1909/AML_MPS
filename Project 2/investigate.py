import biosppy.signals.ecg as ecg
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
import scipy
from scipy.signal import welch

sampling_rate = 300


def load_data():
    # Load the data
    train = pd.read_csv('Data/train.csv')
    X_test = pd.read_csv('Data/test.csv')

    y_train = train['y'].to_numpy()
    X_train = train.drop(columns=['y', 'id'])
    return X_train, y_train, X_test


def plot_ecg(x1, y, x2):
    # Plot the ECG signal
    signal = x1.loc[0].dropna().to_numpy(dtype=np.float32)
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=True)
    # plt.plot(signal)
    # plt.show()

def extract_features(signal):
    # Extract features from the ECG signal
    signal = signal.dropna().to_numpy(dtype=np.float32)
    rpeaks = ecg.engzee_segmenter(signal=signal, sampling_rate=sampling_rate)['rpeaks']
    beats = ecg.extract_heartbeats(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate)['templates']

    print(beats.shape)
    print(extract_features_from_heartbeat(beats, sampling_rate).shape)

def extract_intervals_from_heartbeat(heartbeat, r_peaks, sampling_rate):
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)



    q_wave_idx = np.argmin(heartbeat[:len(heartbeat) // 2])
    s_wave_idx = len(heartbeat) // 2 + np.argmin(heartbeat[len(heartbeat) // 2:])
    t_wave_end = s_wave_idx + int(0.2 * sampling_rate)

    qrs_duration = (s_wave_idx - q_wave_idx) / sampling_rate
    qt_interval = (t_wave_end - q_wave_idx) / sampling_rate
    pr_interval = q_wave_idx / sampling_rate


    intervals = np.array([
        mean_rr, std_rr, qrs_duration, qt_interval, pr_interval
    ])

    return intervals

def extract_temporal_features(signal):
    """
    Extract temporal features from the entire ECG signal.

    Parameters:
    - signal: A 1D NumPy array representing the ECG signal

    Returns:
    - A 1D NumPy array containing temporal features
    """
    max_amplitude = np.max(signal)
    min_amplitude = np.min(signal)
    mean_value = np.mean(signal)
    std_value = np.std(signal)
    median_value = np.median(signal)
    energy = np.sum(signal ** 2)
    entropy_value = entropy(signal)
    kurtosis_value = kurtosis(signal)
    skewness = skew(signal)
    rms = np.sqrt(np.mean(signal ** 2))

    signal_range = max_amplitude - min_amplitude

    return np.array([
        max_amplitude, min_amplitude, mean_value, std_value,
        median_value, energy, entropy_value,
        skewness, kurtosis_value, signal_range, rms
    ])


def extract_frequency_features(signal, sampling_rate):
    """
    Extract frequency-domain features using Welch's method.

    Parameters:
    - signal: A 1D NumPy array representing the ECG signal
    - sampling_rate: Sampling rate in Hz

    Returns:
    - A 1D NumPy array containing frequency-domain features
    """
    # Step 1: Calculate Power Spectral Density (PSD) using Welch's method
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=len(signal) // 2)

    # Step 2: Define frequency bands for LF and HF
    lf_band = (0.04, 0.15)  # Low-Frequency band (0.04 Hz to 0.15 Hz)
    hf_band = (0.15, 0.4)  # High-Frequency band (0.15 Hz to 0.4 Hz)

    # Extract LF and HF power
    lf_indices = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_indices = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

    lf_power = np.trapz(psd[lf_indices], freqs[lf_indices])
    hf_power = np.trapz(psd[hf_indices], freqs[hf_indices])

    # Total power
    total_power = np.trapz(psd, freqs)

    # Dominant frequency
    dominant_frequency = freqs[np.argmax(psd)]

    spectral_entropy = entropy(psd)
    spectral_centroid = np.sum(freqs, psd) / np.sum(psd)
    # Spectral Bandwidth
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

    # Spectral Rolloff (85% energy)
    cumulative_energy = np.cumsum(psd)
    rolloff_threshold = 0.85 * cumulative_energy[-1]
    spectral_rolloff = freqs[np.where(cumulative_energy >= rolloff_threshold)[0][0]]

    # Spectral Flatness
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)

    # Calculate LF/HF ratio
    lf_hf_ratio = lf_power / (hf_power + 1e-12)

    return np.array([
        lf_power, hf_power, total_power,
        dominant_frequency, lf_hf_ratio, spectral_entropy,
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        spectral_flatness
    ])

def calculate_representative_heartbeats(templates):
    """
    Calculate the mean, standard deviation, and median heartbeats.

    Parameters:
    - templates: A 2D NumPy array of shape (num_beats, num_samples_per_beat)

    Returns:
    - mean_heartbeat: The mean heartbeat
    - std_heartbeat: The standard deviation of the heartbeats
    - median_heartbeat: The median heartbeat
    """
    mean_heartbeat = np.mean(templates, axis=0)
    std_heartbeat = np.std(templates, axis=0)
    median_heartbeat = np.median(templates, axis=0)
    return mean_heartbeat, std_heartbeat, median_heartbeat


def extract_features_from_heartbeat(heartbeat, sampling_rate):
    """
    Extract both time-domain and frequency-domain features from the mean heartbeat.

    Parameters:
    - mean_heartbeat: A 1D NumPy array representing the mean heartbeat
    - sampling_rate: Sampling rate in Hz

    Returns:
    - A 1D NumPy array containing the extracted features
    """
    num_samples = len(heartbeat)

    # Step 1: Extract time-domain features
    max_amplitude = np.max(heartbeat)
    min_amplitude = np.min(heartbeat)
    mean_value = np.mean(heartbeat)
    std_value = np.std(heartbeat)
    median_value = np.median(heartbeat)
    energy = np.sum(heartbeat ** 2)
    entropy_value = entropy(heartbeat)
    kurtosis_value = kurtosis(heartbeat)
    rms = np.sqrt(np.mean(heartbeat ** 2))
    skewness = skew(heartbeat)

    # Step 2: Extract frequency-domain features using Welch's method
    freqs, psd = welch(heartbeat, fs=sampling_rate, nperseg=num_samples)

    spectral_entropy = entropy(psd)
    total_power = np.trapz(psd, freqs)
    spectral_centroid = np.sum(freqs, psd) / np.sum(psd)
    # Spectral Bandwidth
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

    # Spectral Rolloff (85% energy)
    cumulative_energy = np.cumsum(psd)
    rolloff_threshold = 0.85 * cumulative_energy[-1]
    spectral_rolloff = freqs[np.where(cumulative_energy >= rolloff_threshold)[0][0]]

    # Spectral Flatness
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)
    # Define frequency bands for LF and HF
    lf_band = (0.04, 0.15)  # Low-Frequency band (0.04 Hz to 0.15 Hz)
    hf_band = (0.15, 0.4)  # High-Frequency band (0.15 Hz to 0.4 Hz)

    # Extract LF and HF power
    lf_indices = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_indices = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    lf_power = np.trapz(psd[lf_indices], freqs[lf_indices])
    hf_power = np.trapz(psd[hf_indices], freqs[hf_indices])

    # Calculate LF/HF ratio
    lf_hf_ratio = lf_power / (hf_power + 1e-12)

    # Combine all features into a single feature vector
    features = np.array([
        max_amplitude, min_amplitude, mean_value, std_value,
        median_value, energy, entropy_value, kurtosis_value, skewness, rms,
        lf_power, hf_power, lf_hf_ratio, spectral_entropy, total_power,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    ])

    return features

if __name__ == "__main__":
    x1, y, x2 = load_data()

    extract_features(x1.loc[0])
