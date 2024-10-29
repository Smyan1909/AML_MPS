import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
def normalize_features(data):
    """
    Normalize the features of the data set
    :param data: data set
    :return: normalized data set
    """
    # Create a StandardScaler object
    scaler = StandardScaler()

    normalized_data = scaler.fit_transform(data)

    return normalized_data, scaler
def detect_outliers_PCA_GMM(standardized_features, n_clusters=2, threshold_percentile=2.5):
    """
    Detect outliers in the dataset using PCA for dimensionality reduction and GMM for clustering.

    :param standardized_features: The standardized features of the dataset.
    :param n_clusters: The number of clusters to form (default is 2).
    :param threshold_percentile: The percentile to determine the threshold for outlier detection (default is 2.5).
    :return: A boolean array indicating which samples are outliers.
    """

    scaler = RobustScaler()

    standardized_features = scaler.fit_transform(standardized_features)

    pca = PCA(n_components=0.95)  # retain 95% of the variance
    reduced_features = pca.fit_transform(standardized_features)
    # Check how many components were retained
    # print(f"Number of components retained: {pca.n_components_}")

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(reduced_features)

    scores = gmm.score_samples(reduced_features)
    threshold = np.percentile(scores, threshold_percentile)

    anomalies = scores < threshold

    return anomalies


def detect_outliers_IsolationForest(standardized_features, contamination=0.025):
    """
    Detect outliers in the dataset using Isolation Forest.
    :param standardized_features: The standardized features of the dataset.
    :param contamination: The amount of contamination of the data set (default is 0.01).
    :return: A boolean array indicating which samples are outliers.
    """

    scaler = RobustScaler()

    standardized_features = scaler.fit_transform(standardized_features)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(standardized_features)

    predictions = iso_forest.predict(standardized_features)

    anomalies = predictions == -1

    return anomalies

def detect_outliers_Autoencoders(standardized_features, encoding_dim=32, epochs=50, batch_size=64):
    """
    Detect outliers in the dataset using Autoencoders.
    :param standardized_features: The standardized features of the dataset.
    :param encoding_dim: The size of the encoded representation (default is 32).
    :param epochs: The number of epochs for training (default is 100).
    :param batch_size: The batch size for training (default is 128).
    :return: A boolean array indicating which samples are outliers.
    """

    scaler = RobustScaler()

    standardized_features = scaler.fit_transform(standardized_features)

    # Create an input layer
    input_layer = Input(shape=(standardized_features.shape[1],))

    # Create the encoder
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)

    # Create the decoder
    decoded = Dense(standardized_features.shape[1], activation='sigmoid')(encoded)

    # Create the autoencoder
    autoencoder = Model(input_layer, decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the autoencoder
    autoencoder.fit(standardized_features, standardized_features,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0)

    # Predict the features
    predicted_features = autoencoder.predict(standardized_features)

    # Calculate the mean squared error
    mse = np.mean(np.power(standardized_features - predicted_features, 2), axis=1)

    # Calculate the threshold
    threshold = np.percentile(mse, 97.5)

    # Detect the outliers
    anomalies = mse > threshold

    return anomalies


def replace_NaN(data, n_neighbors=5):
    """
    Replace NaN values in the data set using KNN imputation.
    :param data:
    :param n_neighbors:
    :return: DataFrame with filled NaN values using KNN imputation.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)

    filled_data = imputer.fit_transform(data)

    return filled_data, imputer


def impute_median(data):
    """
    :param data:
    :return: array filled with imputed data using median values
    """

    imputer = SimpleImputer(strategy="median")

    filled_data = imputer.fit_transform(data)

    return filled_data, imputer



if __name__ == "__main__":

    data = pd.read_csv("Data\X_train.csv")

    data.drop('id', axis=1, inplace=True)

    X_train_no_NaN = replace_NaN(data, 5)

    # Compare two dataframes element-wise
    comparison = data == X_train_no_NaN

    # Count the number of elements that are exactly the same
    num_same_elements = comparison.sum().sum()

    total_nan = data.isna().sum().sum()

    total_vals = data.shape[0] * data.shape[1]

    print(f"Total NaN values in the DataFrame: {total_nan}")
    print(f"Number of elements that are exactly the same: {num_same_elements}")
    print(f"Total number of values in the DataFrame: {total_vals}")