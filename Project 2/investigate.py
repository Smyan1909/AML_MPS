import biosppy.signals.ecg as ecg
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis, uniform, randint
import scipy
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

sampling_rate = 300


def load_data():
    # Load the data
    train = pd.read_csv('Data/train.csv')
    X_test = pd.read_csv('Data/test.csv')

    y_train = train['y'].to_numpy().ravel()
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

    mean_heartbeat, std_heartbeat, median_heartbeat = calculate_representative_heartbeats(beats)

    mean_heartbeat_features = extract_features_from_heartbeat(mean_heartbeat, sampling_rate)
    std_heartbeat_features = extract_features_from_heartbeat(std_heartbeat, sampling_rate)
    median_heartbeat_features = extract_features_from_heartbeat(median_heartbeat, sampling_rate)

    temporal_features = extract_temporal_features(signal)
    frequency_features = extract_frequency_features(signal, sampling_rate)
    intervals = extract_intervals_from_heartbeat(mean_heartbeat, rpeaks, sampling_rate)

    features = np.hstack([temporal_features, frequency_features, mean_heartbeat_features, std_heartbeat_features, median_heartbeat_features, intervals])

    return features

def create_features(X):
    features = []
    for i in range(len(X)):
        signal = X.loc[i]
        features.append(extract_features(signal))
        print(f'Extracted features from signal {i + 1}/{len(X)}')

    features = np.array(features)
    features_df = pd.DataFrame(features)
    features_df.to_csv('Data/test_features.csv', index=False)

def extract_intervals_from_heartbeat(heartbeat, r_peaks, sampling_rate):
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)

    if not isinstance(heartbeat, np.ndarray) or heartbeat.size == 0:
        print("Error: Heartbeat input is not a valid array or is empty.")
        return np.array([mean_rr, std_rr, np.nan, np.nan, np.nan])

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
    if not isinstance(heartbeat, np.ndarray) or heartbeat.size == 0:
        print("Error: Heartbeat input is not a valid array or is empty.")
        return np.array([np.nan] * 18)

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

    spectral_entropy = entropy(psd)
    total_power = np.trapz(psd, freqs)
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
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    ])

    return features

def load_features():
    return pd.read_csv('Data/train_features.csv')

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

    return X_scaled

def select_features(features, y):
    selector = Pipeline([
        ('threshold', VarianceThreshold(0.01)),
    ])
    X_new = selector.fit_transform(features, y)

    return X_new

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

        # Step 1: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 2: Feature selection
        selector = Pipeline([
            ('threshold', VarianceThreshold(0.01)),
        ])
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        # Step 3: Train a Classifier
        model = HistGradientBoostingClassifier(learning_rate=0.15, max_depth=7, max_iter=350, max_leaf_nodes=30,
                                               l2_regularization=0.7, max_bins=140, early_stopping=False,
                                               min_samples_leaf=30, random_state=42)
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
        ('selector', VarianceThreshold(0.01)),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])

    # Define the parameter grid
    param_distributions = {
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_iter': randint(100, 500),
        'model__max_depth': [3, 5, 7, 9, None],
        'model__min_samples_leaf': randint(1, 50),
        'model__max_leaf_nodes': randint(10, 100),
        'model__l2_regularization': uniform(0.0, 1.0),
        'model__max_bins': randint(128, 256),
        'model__early_stopping': [True, False]
    }

    # Define the cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=200,  # Number of parameter settings to try
        scoring='f1_weighted',
        n_jobs=-1,  # Use all available processors
        cv=skf,
        verbose=2,
        random_state=42
    )

    # Run the randomized search
    search.fit(X, y)

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation F1 score: {search.best_score_:.4f}")

    return search.best_estimator_

def create_submission_first(X, y, X_test):
    # Pre-process the data
    X_scaled = scale_features(X)
    X_test_scaled = scale_features(X_test)
    X_selected = select_features(X_scaled, y)
    X_test_selected = select_features(X_test_scaled, y)

    # Train the model
    model = HistGradientBoostingClassifier(learning_rate=0.15, max_depth=7, max_iter=350, max_leaf_nodes=30,
                                           l2_regularization=0.7, max_bins=140, early_stopping=False,
                                           min_samples_leaf=30, random_state=42)
    model.fit(X_selected, y)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Create a submission file
    submission = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred})
    submission.to_csv('Data/submission.csv', index=False)

if __name__ == "__main__":
    x1, y, x2 = load_data()

    create_features(x2)
    #create_features(x1)
    #create_train_labels(y)

    #features = load_features()
    #y = load_labels()
    #features, y = remove_NaNs(features, y)
    #cross_validate_hist_grad_boost(features, y, n_splits=5)
    #tune_hist_gradient_boosting(features, y)
