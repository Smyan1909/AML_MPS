from preprocessing import *
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
def remove_outliers(X, y=None):
    """
    Remove outliers from the data set

    :param data: data set
    :return: data set without outliers
    """
    outliers = detect_outliers_IsolationForest(X, n_estimators=400, contamination=0.045) # 0.075


    if y is not None:
        return X[~outliers], y[~outliers], outliers
    else:
        return X[~outliers], outliers


def load_data():
    """
    Load the data set

    :return: data set
    """
    X_train = pd.read_csv('Data\X_train.csv').drop(columns=['id']).to_numpy()
    y_train = pd.read_csv('Data\y_train.csv').drop(columns=['id']).to_numpy()
    X_test = pd.read_csv('Data\X_test.csv').drop(columns=['id']).to_numpy()


    return X_train, y_train, X_test


def visualize_outliers(X, X_test, anomalies):
    """
    Visualize the outliers

    :param X: data set
    :param anomalies: boolean array indicating the outliers
    """
    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    X_test = pca.transform(imputer.transform(X_test))
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', marker='o')
    plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomaly', marker='x')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='green', label='Test', marker='s')
    plt.legend()
    plt.show()


def process_features(X, is_train=True, imputer=None, scaler=None):
    """
    Impute missing values and normalize features.

    :param X: Input data
    :param is_train: Whether this is training data (to fit imputer and scaler)
    :param imputer: Pre-fitted imputer for test data
    :param scaler: Pre-fitted scaler for test data
    :return: Processed X, imputer, scaler
    """
    if is_train:
        # Imputation
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled, imputer, scaler
    else:
        # Use the fitted imputer and scaler for test data
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        return X_scaled

def create_grouped_kfold_with_clustering(X_train, y_train, X_test, n_splits=5, n_clusters=5):
    """
    Create grouped K-Fold cross-validation splits by clustering training and test data together.

    :param X_train: Training feature matrix
    :param y_train: Training target variable
    :param X_test: Test feature matrix
    :param n_splits: Number of K-Folds
    :param n_clusters: Number of clusters for grouping
    :return: List of tuples with (train_index, val_index) for each fold
    """
    # Step 1: Combine training and test data temporarily for clustering
    combined_data = np.vstack((X_train, X_test))

    # Step 2: Apply KMeans clustering to group similar samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(combined_data)

    # Step 3: Separate the cluster labels for training data only
    train_groups = cluster_labels[:len(X_train)]  # Use only training cluster labels

    # Step 4: Use GroupKFold with the cluster labels as groups
    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    for train_index, val_index in gkf.split(X_train, y_train, groups=train_groups):
        folds.append((train_index, val_index))

    return folds


def visualize_clustering(X_train, X_test, n_clusters=5, method='pca'):
    """
    Visualize the clustering of combined training and test data.

    :param X_train: Training feature matrix
    :param X_test: Test feature matrix
    :param n_clusters: Number of clusters to create
    :param method: Dimensionality reduction method ('pca' or 'tsne')
    """
    # Combine training and test data for clustering
    combined_data = np.vstack((X_train, X_test))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(combined_data)

    # Separate cluster labels for training and test data
    train_labels = cluster_labels[:len(X_train)]
    test_labels = cluster_labels[len(X_train):]

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be 'pca' or 'tsne'")

    reduced_data = reducer.fit_transform(combined_data)

    # Separate reduced data for training and test sets
    reduced_train = reduced_data[:len(X_train)]
    reduced_test = reduced_data[len(X_train):]

    # Plot the clusters
    plt.figure(figsize=(10, 6))

    # Plot training data points with cluster labels
    scatter = plt.scatter(reduced_train[:, 0], reduced_train[:, 1],
                          c=train_labels, cmap='tab10', marker='o', label='Train')

    # Plot test data points with cluster labels
    plt.scatter(reduced_test[:, 0], reduced_test[:, 1],
                c=test_labels, cmap='tab10', marker='s', edgecolor='k', label='Test')

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)
    plt.legend(['Train', 'Test'])

    plt.title(f'Clustering Visualization with {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
def check_anomaly_removal():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    visualize_outliers(X_train, X_test,outliers)

def check_clustering():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    visualize_clustering(X_train=X, X_test=X_test, n_clusters=5, method='pca')

if __name__ == "__main__":
    check_clustering()
    #check_anomaly_removal()