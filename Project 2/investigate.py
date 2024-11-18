import biosppy.signals.ecg as ecg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy, skew, kurtosis, uniform, randint
import scipy
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.signal import butter, filtfilt
import antropy as ant
import neurokit2 as nk
import pywt

sampling_rate = 300


def load_data():
    # Load the data
    train = pd.read_csv('Data/train.csv')
    X_test = pd.read_csv('Data/test.csv')

    y_train = train['y'].to_numpy().ravel()
    X_train = train.drop(columns=['y', 'id'])
    return X_train, y_train, X_test


def plot_ecg(x1, index):
    # Plot the ECG signal
    signal = x1.loc[index].dropna().to_numpy(dtype=np.float32)

    signal = bandpass_filter(signal, highcut=45)[180: -300]

    if np.abs(np.min(signal)) > np.max(signal):
        signal *= -1
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=True)

    plt.figure(2)
    plt.plot(np.mean(out['templates'], axis=0))

    print("Signal Length: ", len(np.mean(out['templates'], axis=0)))

    plt.show()


def bandpass_filter(signal, lowcut=0.5, highcut=40, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
    - signal: 1D numpy array of the ECG signal.
    - lowcut: Lower cutoff frequency (in Hz).
    - highcut: Upper cutoff frequency (in Hz).
    - sampling_rate: Sampling rate of the signal (in Hz).
    - order: Order of the filter.

    Returns:
    - filtered_signal: The bandpass filtered signal.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
def extract_autocorrelation(signal, max_lag=50):
    """
    Calculate the autocorrelation of a signal up to a specified lag.

    Parameters:
    - signal: 1D NumPy array representing the ECG signal.
    - max_lag: Maximum number of lags to compute the autocorrelation for.

    Returns:
    - autocorr_features: A 1D NumPy array of autocorrelation values up to max_lag.
    """
    # Remove mean from the signal
    signal = signal - np.mean(signal)

    # Compute the full autocorrelation
    autocorr_full = np.correlate(signal, signal, mode='full')

    # Extract the second half (positive lags)
    autocorr = autocorr_full[len(autocorr_full) // 2:]

    # Normalize the autocorrelation
    autocorr = autocorr / autocorr[0]

    # Select only up to the specified maximum lag
    autocorr_features = autocorr[:max_lag]

    return autocorr_features

def extract_features(signal):
    # Extract features from the ECG signal
    signal = signal.dropna().to_numpy(dtype=np.float32)

    signal = bandpass_filter(signal, highcut=45)[180: -300]

    if np.abs(np.min(signal)) > np.max(signal):
        signal *= -1

    ecg_out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    rpeaks = ecg_out['rpeaks']
    beats = ecg_out['templates']
    hr = ecg_out['heart_rate']

    nk_signal, info = nk.ecg_process(ecg_out['filtered'], sampling_rate=sampling_rate)

    q_pos = np.array(info['ECG_Q_Peaks'])
    s_pos = np.array(info['ECG_S_Peaks'])
    t_pos = np.array(info['ECG_T_Peaks'])


    mean_heartbeat, std_heartbeat, median_heartbeat = calculate_representative_heartbeats(beats)

    mean_heartbeat_features = extract_features_from_heartbeat(mean_heartbeat, sampling_rate)
    std_heartbeat_features = extract_features_from_heartbeat(std_heartbeat, sampling_rate)
    median_heartbeat_features = extract_features_from_heartbeat(median_heartbeat, sampling_rate)

    temporal_features = extract_temporal_features(signal)
    frequency_features = extract_frequency_features(signal, sampling_rate)
    #intervals = extract_intervals_from_heartbeat(mean_heartbeat, rpeaks, sampling_rate)
    intervals = extract_intervals_from_points(rpeaks, q_pos, s_pos, t_pos, sampling_rate)

    hrv_features = compute_hrv_features(np.diff(rpeaks) / sampling_rate)

    features = np.hstack([temporal_features, frequency_features, mean_heartbeat_features, std_heartbeat_features,
                          median_heartbeat_features, intervals, mean_heartbeat, std_heartbeat, hrv_features,
                          np.mean(hr), np.std(hr), np.median(hr)])

    return features

def create_features(X, filename='Data/train_features.csv'):
    features = []
    for i in range(len(X)):
        signal = X.loc[i]
        features.append(extract_features(signal))
        print(f'Extracted features from signal {i + 1}/{len(X)}')

    features = np.array(features)
    features_df = pd.DataFrame(features)
    features_df.to_csv(filename, index=False)

def compute_hrv_features(rr_intervals):
    # RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    # SDNN
    sdnn = np.std(rr_intervals)
    # PNN50
    pnn50 = np.sum(np.diff(rr_intervals) > 0.05) / len(rr_intervals) * 100
    return rmssd, sdnn, pnn50

def extract_intervals_from_points(r_peaks, t_vals, q_vals, s_vals, sampling_rate):
    t_vals = t_vals[~np.isnan(t_vals)]
    q_vals = q_vals[~np.isnan(q_vals)]
    s_vals = s_vals[~np.isnan(s_vals)]
    min_length = min(len(q_vals), len(s_vals), len(t_vals))
    q_vals = q_vals[:min_length]
    s_vals = s_vals[:min_length]
    t_vals = t_vals[:min_length]

    rr_intervals = np.diff(r_peaks) / sampling_rate
    mean_rr = np.nanmean(rr_intervals)
    std_rr = np.std(rr_intervals)
    median_rr = np.median(rr_intervals)

    qrs_duration = (s_vals - q_vals) / sampling_rate
    mean_qrs = np.nanmean(qrs_duration)
    std_qrs = np.nanstd(qrs_duration)
    median_qrs = np.nanmedian(qrs_duration)

    qt_interval = (t_vals - q_vals) / sampling_rate
    mean_qt = np.nanmean(qt_interval)
    std_qt = np.nanstd(qt_interval)
    median_qt = np.nanmedian(qt_interval)

    qq_interval = np.diff(q_vals) / sampling_rate
    mean_qq = np.nanmean(qq_interval)
    std_qq = np.nanstd(qq_interval)
    median_qq = np.nanmedian(qq_interval)

    ss_interval = np.diff(s_vals) / sampling_rate
    mean_ss = np.nanmean(ss_interval)
    std_ss = np.nanstd(ss_interval)
    median_ss = np.nanmedian(ss_interval)

    tt_interval = np.diff(t_vals) / sampling_rate
    mean_tt = np.nanmean(tt_interval)
    std_tt = np.nanstd(tt_interval)
    median_tt = np.nanmedian(tt_interval)

    intervals = np.array([
        mean_rr, std_rr, median_rr,
        mean_qrs, std_qrs, median_qrs,
        mean_qt, std_qt, median_qt,
        mean_qq, std_qq, median_qq,
        mean_ss, std_ss, median_ss,
        mean_tt, std_tt, median_tt
    ])

    return intervals

def extract_intervals_from_heartbeat(heartbeat, r_peaks, sampling_rate):
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    r_wave_idx = np.argmax(heartbeat)

    #if not isinstance(heartbeat, np.ndarray) or heartbeat.size == 0:
    #    print("Error: Heartbeat input is not a valid array or is empty.")
    #    return np.array([mean_rr, std_rr, np.nan, np.nan, np.nan])

    q_wave_idx = np.argmin(heartbeat[:len(heartbeat) // 2])
    s_wave_idx = len(heartbeat) // 2 + np.argmin(heartbeat[len(heartbeat) // 2:])
    t_wave_end = s_wave_idx + int(0.2 * sampling_rate)

    qrs_duration = (s_wave_idx - q_wave_idx) / sampling_rate
    qt_interval = (t_wave_end - q_wave_idx) / sampling_rate
    pr_interval = q_wave_idx / sampling_rate


    intervals = np.array([
        mean_rr, std_rr, qrs_duration, qt_interval, pr_interval, q_wave_idx, s_wave_idx, t_wave_end, r_wave_idx
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
    kurtosis_value = kurtosis(signal)
    skewness = skew(signal)
    rms = np.sqrt(np.mean(signal ** 2))

    signal_range = max_amplitude - min_amplitude

    return np.array([
        max_amplitude, min_amplitude, mean_value, std_value,
        median_value, energy,
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

    mean_power = np.mean(psd)
    std_power = np.std(psd)
    median_power = np.median(psd)

    # Dominant frequency
    dominant_frequency = freqs[np.argmax(psd)]

    spectral_entropy = entropy(psd)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
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
        spectral_flatness, mean_power, std_power, median_power
    ])

def extract_wavelet_features(signal):
    # Perform wavelet decomposition using 'db4' wavelet
    coeffs = pywt.wavedec(signal, 'db4', level=5)

    # Extract features from the coefficients
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    features = [
        np.mean(cA5), np.std(cA5), np.sum(np.abs(cA5)),
        np.mean(cD5), np.std(cD5), np.sum(np.abs(cD5)),
        np.mean(cD4), np.std(cD4), np.sum(np.abs(cD4)),
        np.mean(cD3), np.std(cD3), np.sum(np.abs(cD3))
    ]
    return np.array(features)


def extract_nonlinear_features(signal):
    sample_entropy = ant.sample_entropy(signal)
    approximate_entropy = ant.app_entropy(signal)
    return np.array([sample_entropy, approximate_entropy])

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
    kurtosis_value = kurtosis(heartbeat)
    rms = np.sqrt(np.mean(heartbeat ** 2))
    skewness = skew(heartbeat)

    # Step 2: Extract frequency-domain features using Welch's method
    freqs, psd = welch(heartbeat, fs=sampling_rate, nperseg=num_samples)

    mean_power = np.mean(psd)
    std_power = np.std(psd)
    median_power = np.median(psd)
    spectral_entropy = entropy(psd)
    total_power = np.trapz(psd, freqs)
    dominant_frequency = freqs[np.argmax(psd)]
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
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
        median_value, energy, kurtosis_value, skewness, rms,
        lf_power, hf_power, lf_hf_ratio, spectral_entropy, total_power,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, mean_power, std_power, median_power,
        dominant_frequency
    ])

    return features

def load_features(filename='Data/train_features.csv'):
    return pd.read_csv(filename).to_numpy()

def load_labels():
    return pd.read_csv('Data/train_labels.csv').to_numpy().ravel()

def remove_NaNs(features, y):
    # Pre-process the data
    initial_rows = features.shape[0]

    print(f'Initial number of rows: {initial_rows}')

    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    X = features.dropna()
    removed_rows = set(features.index) - set(X.index)

    print(f'Removed {len(removed_rows)} rows with NaN values')

    y = np.delete(y, list(removed_rows))

    return X.to_numpy(dtype=np.float32), y

def scale_features(features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled, scaler

def select_features(features, y):
    selector = Pipeline([
        ('threshold', VarianceThreshold(0.01)),
        ('selector', SelectKBest(f_classif, k=452))
    ])
    X_new = selector.fit_transform(features, y)

    return X_new, selector


def filter_correlated_features(X, correlation_threshold=0.9):
    """
    Removes features that are highly correlated with each other.

    Parameters:
    - X: 2D NumPy array (features)
    - correlation_threshold: The threshold above which features are considered highly correlated (default is 0.9).

    Returns:
    - X_filtered: A 2D NumPy array with highly correlated features removed.
    - to_drop: List of indices of the dropped features.
    """
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)
    corr_matrix = np.abs(corr_matrix)  # Use absolute values for correlations

    # Get the indices of the upper triangle of the correlation matrix
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Identify features to drop based on the correlation threshold
    to_drop = [col for col in range(corr_matrix.shape[1]) if
               any(corr_matrix[col, upper_triangle[:, col]] > correlation_threshold)]

    # Drop the highly correlated features
    X_filtered = np.delete(X, to_drop, axis=1)

    return X_filtered, to_drop

def feature_selection_by_anova(X, y, p_value_threshold=0.05):
    """
    Perform feature selection using ANOVA F-test and remove features with p-values > 0.05.

    Parameters:
    - X: Feature matrix (NumPy array or DataFrame)
    - y: Target vector (NumPy array or Series)
    - p_value_threshold: The threshold for p-value (default is 0.05)

    Returns:
    - X_selected: Feature matrix with selected features
    - selected_indices: List of indices of the selected features
    """
    # Perform ANOVA F-test

    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)

    f_values, p_values = f_classif(X, y)

    # Filter features based on p-value threshold
    selected_indices = np.where(p_values <= p_value_threshold)[0]

    # Select features with p-values <= threshold
    X_selected = X[:, selected_indices]

    print(f"Selected {len(selected_indices)} features out of {X.shape[1]}")

    return X_selected, selected_indices

def create_train_labels(y):
    y = pd.DataFrame(y)

    y.to_csv('Data/train_labels.csv', index=False)

def cross_validate_hist_grad_boost(X, y, n_splits=5):
    """
    Perform cross-validation using a Naive Bayes classifier.
    Preprocess, scale, and select features within each fold to avoid data leakage.
    """
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nProcessing fold {fold + 1}/{n_splits}...")

        # Split into training and validation sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        imputer = KNNImputer(n_neighbors=5)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Step 1: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)



        # Step 2: Feature selection
        selector = Pipeline([
            ('variance', VarianceThreshold(0.01)),
            ('selector', SelectKBest(f_classif, k=452))
        ])
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        # {'model__validation_fraction': 0.1, 'model__scoring': 'f1_weighted', 'model__min_samples_leaf': 40, 'model__max_leaf_nodes': 31, 'model__max_iter': 1000, 'model__max_depth': 9, 'model__max_bins': 128, 'model__learning_rate': 0.05, 'model__l2_regularization': 1.0, 'model__early_stopping': False}
        # Step 3: Train a Classifier
        model = HistGradientBoostingClassifier(learning_rate=0.15, max_depth=7, max_iter=350, max_leaf_nodes=30,
                                               l2_regularization=0.7, max_bins=140, early_stopping=False,
                                               min_samples_leaf=30, random_state=42) # F1: 0.7874

        model.fit(X_train_selected, y_train)

        # Step 4: Make predictions on the validation set
        y_pred = model.predict(X_test_selected)
        # Step 5: Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store metrics
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

        print(
            f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Print average metrics
    print("\nCross-validation results:")
    print(f"Mean Accuracy: {np.mean(metrics['accuracy']):.4f}")
    print(f"Mean Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Mean Recall: {np.mean(metrics['recall']):.4f}")
    print(f"Mean F1 Score: {np.mean(metrics['f1']):.4f}")

    return metrics


def tune_hist_gradient_boosting(X, y):
    """
    Tune hyperparameters for HistGradientBoostingClassifier using RandomizedSearchCV.
    """
    # Define the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('variance', VarianceThreshold(0.01)),
        ('selector', SelectKBest(f_classif, k=452)),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])

    # Define the parameter grid
    param_distribution = {
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Step size at each iteration
        'model__max_iter': [100, 200, 300, 500, 1000],  # Number of boosting iterations
        'model__max_depth': [3, 5, 7, 9, None],  # Depth of each tree
        'model__min_samples_leaf': [10, 20, 30, 40, 50],  # Minimum samples required in a leaf
        'model__max_leaf_nodes': [15, 31, 63, 127],  # Maximum leaves per tree
        'model__l2_regularization': [0.0, 0.1, 0.5, 1.0],  # L2 regularization term
        'model__max_bins': [128, 256, 512],  # Number of bins for continuous features
        'model__early_stopping': [True, False],  # Early stopping to avoid overfitting
        'model__validation_fraction': [0.1, 0.15, 0.2],  # Fraction of data for validation
        'model__scoring': ['f1_weighted', 'accuracy', 'roc_auc']  # Scoring metric to optimize
    }
    # Define the cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distribution,
        n_iter=50,  # Number of parameter settings to try
        scoring='f1_weighted',
        n_jobs=-1,  # Use all available processors
        cv=skf,
        verbose=3,
        random_state=42
    )

    # Run the randomized search
    search.fit(X, y)

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation F1 score: {search.best_score_:.4f}")

    return search.best_estimator_

def create_submission(X, y, X_test):
    # Pre-process the data

    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)

    X_scaled, scaler = scale_features(X)
    X_test_scaled = scaler.transform(X_test)

    X_selected, selector = select_features(X_scaled, y)
    X_test_selected = selector.transform(X_test_scaled)

    # Train the model
    model = HistGradientBoostingClassifier(learning_rate=0.15, max_depth=7, max_iter=350, max_leaf_nodes=30,
                                           l2_regularization=0.7, max_bins=140, early_stopping=False,
                                           min_samples_leaf=30, random_state=42)
    model.fit(X_selected, y)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Create a submission file
    submission = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred})
    submission.to_csv('Data/submission2_ss.csv', index=False)

if __name__ == "__main__":

    #x1, y, x2 = load_data()

    #plot_ecg(x2, 2872)
    #create_features(x1, filename='Data/train_features_with_ecg_feats.csv')
    #create_features(x2, filename='Data/test_features_with_ecg_feats.csv')
    #create_train_labels(y)

    features = load_features(filename='Data/train_features_with_ecg_feats.csv')
    #X_test = load_features(filename='Data/test_features_with_ecg_feats.csv')

    y = load_labels()
    #create_submission(features, y, X_test)
    #feature_selection_by_anova(features, y)
    #features, y = remove_NaNs(features, y)
    #cross_validate_hist_grad_boost(features, y, n_splits=10)
    tune_hist_gradient_boosting(features, y)
