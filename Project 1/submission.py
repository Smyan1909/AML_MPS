from preprocessing import *
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel as C
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import seaborn as sns
import numpy as np


def feature_selection(X, y):


    selector = Pipeline([
        ('variance', VarianceThreshold(0.01)),
        ('univariate', SelectKBest(f_regression, k=100)),
    ])

    X_selected = selector.fit_transform(X, y)

    return X_selected, selector


def remove_redundant_features(X, correlation_threshold=0.8):
    """
    Remove redundant features based on a correlation threshold.

    :param X: DataFrame or NumPy array of features
    :param correlation_threshold: Threshold above which features are considered redundant
    :return: Reduced feature matrix as NumPy array, and a list of indices of the removed features
    """
    # If X is a NumPy array, convert it to DataFrame for easier column operations
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation above the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    # Get indices of the features to drop
    to_drop_indices = [X.columns.get_loc(col) for col in to_drop]

    # Drop redundant features
    X_reduced = X.drop(columns=to_drop)

    # Convert the reduced DataFrame back to a NumPy array
    X_reduced_np = X_reduced.to_numpy()

    return X_reduced_np, to_drop_indices

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
    y_train = pd.read_csv('Data\y_train.csv').drop(columns=['id']).to_numpy().ravel()
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


def validate_clustered_model_per_cluster_with_training_score(X, y, n_clusters=5, n_splits=5):
    """
    Validate each cluster-specific model using K-Fold within each cluster and print both training and validation R² scores.
    :param X: Full feature matrix
    :param y: Target values
    :param n_clusters: Number of clusters for grouping
    :param n_splits: Number of splits for KFold within each cluster
    :return: List of validation scores for each cluster, and an overall average score
    """
    # Step 1: Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    cluster_scores = []  # Store validation scores per cluster
    training_scores = []  # Store training scores per cluster

    for cluster in range(n_clusters):
        # Step 2: Filter data for the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        X_cluster = X[cluster_indices]
        y_cluster = y[cluster_indices]

        # Step 3: Use K-Fold within this cluster
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cluster_fold_scores = []
        cluster_fold_training_scores = []

        for train_index, val_index in kf.split(X_cluster):
            X_train, X_val = X_cluster[train_index], X_cluster[val_index]
            y_train, y_val = y_cluster[train_index], y_cluster[val_index]

            # Step 4: Train model for this cluster
            models, selectors, _ = train_cluster_models(X_train, y_train, n_clusters=1)  # Only 1 cluster model here

            # Step 5: Transform data with selector and make predictions for validation
            X_val_selected = selectors[0].transform(X_val)
            y_pred_val = models[0].predict(X_val_selected)

            # Step 6: Calculate validation R² score
            val_score = r2_score(y_val, y_pred_val)
            cluster_fold_scores.append(val_score)

            # Step 7: Calculate training R² score
            X_train_selected = selectors[0].transform(X_train)
            y_pred_train = models[0].predict(X_train_selected)
            train_score = r2_score(y_train, y_pred_train)
            cluster_fold_training_scores.append(train_score)

        # Step 8: Average training and validation scores for this cluster
        cluster_avg_val_score = np.mean(cluster_fold_scores)
        cluster_avg_train_score = np.mean(cluster_fold_training_scores)
        cluster_scores.append(cluster_avg_val_score)
        training_scores.append(cluster_avg_train_score)

        print(f"Cluster {cluster} - Average Training R2 Score: {cluster_avg_train_score:.4f}, Average Validation R2 Score: {cluster_avg_val_score:.4f}")

    # Step 9: Overall average scores across all clusters
    overall_val_score = np.mean(cluster_scores)
    overall_train_score = np.mean(training_scores)
    print(f"Overall Average Training R2 Score: {overall_train_score:.4f}")
    print(f"Overall Average Validation R2 Score: {overall_val_score:.4f}")

    return training_scores, cluster_scores, overall_train_score, overall_val_score


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


def train_cluster_models(X_train, y_train, n_clusters=5):
    """
    Train one model per cluster.
    :param X_train: Training feature matrix
    :param y_train: Training target values
    :param n_clusters: Number of clusters
    :return: List of trained models and feature selectors for each cluster
    """
    # Step 1: Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)

    models = []
    selectors = []

    # Step 2: Train one model per cluster
    for cluster in range(n_clusters):
        # Filter data for the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        X_cluster = X_train[cluster_indices]
        y_cluster = y_train[cluster_indices]

        #base_model = Ridge(alpha=1.0)
        X_selected, selector = feature_selection(X_cluster, y_cluster)
        #X_cluster_selected = selector.fit_transform(X_cluster, y_cluster)

        # Train Ridge model on selected features
        model = XGBRegressor(n_estimators=1000, random_state=42)
        model.fit(X_selected, y_cluster)
        # Store the model and selector for each cluster
        models.append(model)
        selectors.append(selector)

    return models, selectors, kmeans


def make_clustered_predictions(X_test, models, selectors, kmeans):
    """
    Make predictions for test data by identifying clusters and using corresponding models.
    :param X_test: Test feature matrix
    :param models: List of trained models for each cluster
    :param selectors: List of feature selectors for each cluster
    :param kmeans: KMeans clustering model
    :return: Array of predictions for the test data
    """
    # Predict the cluster label for each validation sample
    cluster_labels = kmeans.predict(X_test)
    y_pred = np.zeros(X_test.shape[0])

    # Iterate through each cluster and make predictions with the corresponding model
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]

        if cluster_indices.size == 0:
            continue  # Skip if there are no samples for this cluster in the validation set

        # Select the subset of X_val belonging to the current cluster
        X_cluster = X_test[cluster_indices]

        # Apply the feature selector specific to the cluster
        X_selected = selectors[cluster].transform(X_cluster)

        # Predict using the model specific to the cluster
        y_pred[cluster_indices] = models[cluster].predict(X_selected)

    return y_pred
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

def train_evaluate_xgboost(X, y, n_splits=5, random_state=42):
    """
    Train and evaluate an XGBoost model using K-Fold Cross-Validation for regression.

    :param X: Feature matrix
    :param y: Target values
    :param n_splits: Number of folds for cross-validation
    :param random_state: Random state for reproducibility
    :return: Dictionary containing cross-validation scores for R2, MAE, and RMSE
    """
    # Define XGBoost model parameters
    params = {
        'objective': 'reg:squarederror',  # Regression objective
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_lambda': 10,
        'reg_alpha': 0.1,
        'seed': random_state
    }

    # Use K-Fold for cross-validation (5-fold by default)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kfold.split(X, y):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert the data into DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train the model with early stopping
        evals = [(dval, 'validation')]
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50, verbose_eval=False)

        # Predict on validation set
        y_pred = model.predict(dval)

        # Calculate and store metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    # Calculate the mean and std for each metric across folds
    results = {
        'R2 Score': (np.mean(r2_scores), np.std(r2_scores)),
        'Mean Absolute Error': (np.mean(mae_scores), np.std(mae_scores)),
        'Root Mean Squared Error': (np.mean(rmse_scores), np.std(rmse_scores))
    }

    # Print the results
    print("Cross-Validation Performance:")
    for metric, (mean, std) in results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return results

def make_prediction_stacked_ensemble(X, y, X_test, random_state=42):
    params = {
        'objective': 'reg:squarederror',  # Regression objective
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'reg_lambda': 1,
        'reg_alpha': 10,
        'seed': random_state,
        'n_estimators': 300
    }

    # {'model__learning_rate': 0.2, 'model__l2_leaf_reg': 3, 'model__iterations': 700, 'model__depth': 5,
    # 'model__border_count': 128, 'model__bagging_temperature': 1}

    cat_params = {
        'learning_rate': 0.2,
        'l2_leaf_reg': 3,
        'iterations': 700,
        'depth': 5,
        'border_count': 128,
        'bagging_temperature': 1,
        'random_state': random_state,
        'verbose': 0
    }

    # {'model__kernel': 'rbf', 'model__gamma': 'scale', 'model__epsilon': 0.1, 'model__degree': 4, 'model__C': 100}

    svr_params = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'epsilon': 0.1,
        'degree': 4,
        'C': 100
    }

    gpr_params = {
        'kernel': 1 ** 2 * RationalQuadratic(alpha=0.5, length_scale=1),
        'alpha': 1e-05
    }

    X, selector = feature_selection(X, y)
    X_test = selector.transform(X_test)

    xgboost = xgb.XGBRegressor(**params)
    cbr = CatBoostRegressor(**cat_params)
    svr = SVR(**svr_params)
    gpr = GaussianProcessRegressor(**gpr_params)

    model = StackingRegressor(
        estimators=[
            ('gpr', gpr),
            ('cbr', cbr),
            ('svr', svr)
        ],
        final_estimator=Ridge(),
    )

    model.fit(X, y)

    y_pred = model.predict(X_test)

    return y_pred

def train_evaluate_gpr(X, y, n_splits=5, random_state=42):
    # {'model__kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1), 'model__alpha': 1e-05}
    gpr_params = {
        'kernel': 1 ** 2 * RationalQuadratic(alpha=0.5, length_scale=1),
        'alpha': 1e-05
    }

    # Use K-Fold for cross-validation (5-fold by default)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    current_fold = 0

    for train_index, val_index in kfold.split(X, y):
        # Split data

        current_fold += 1
        print(f"Fold {current_fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train, imputer, scaler = process_features(X_train, is_train=True)
        X_val = process_features(X_val, is_train=False, imputer=imputer, scaler=scaler)


        model = GaussianProcessRegressor(**gpr_params)


        # Predict on validation set

        # X_train, indices_to_remove = remove_redundant_features(X_train, correlation_threshold=0.9)
        # X_val = np.delete(X_val, indices_to_remove, axis=1)

        X_train, selector = feature_selection(X_train, y_train)
        X_val = selector.transform(X_val)

        # X_train, indices_to_remove = remove_redundant_features(X_train, correlation_threshold=0.8)
        # X_val = np.delete(X_val, indices_to_remove, axis=1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # Calculate and store metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    # Calculate the mean and std for each metric across folds
    results = {
        'R2 Score': (np.mean(r2_scores), np.std(r2_scores)),
        'Mean Absolute Error': (np.mean(mae_scores), np.std(mae_scores)),
        'Root Mean Squared Error': (np.mean(rmse_scores), np.std(rmse_scores))
    }

    # Print the results
    print("Cross-Validation Performance:")
    for metric, (mean, std) in results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return results

def train_evaluate_stacked_ensemble(X, y, n_splits=5, random_state=42):
    """
        Train and evaluate an XGBoost model using K-Fold Cross-Validation for regression.

        :param X: Feature matrix
        :param y: Target values
        :param n_splits: Number of folds for cross-validation
        :param random_state: Random state for reproducibility
        :return: Dictionary containing cross-validation scores for R2, MAE, and RMSE
        """
    # Define XGBoost model parameters
    # subsample': 0.7, 'reg_lambda': 1, 'reg_alpha': 10, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
    params = {
        'objective': 'reg:squarederror',  # Regression objective
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'reg_lambda': 1,
        'reg_alpha': 10,
        'seed': random_state,
        'n_estimators': 300
    }

    #{'model__learning_rate': 0.2, 'model__l2_leaf_reg': 3, 'model__iterations': 700, 'model__depth': 5,
     #'model__border_count': 128, 'model__bagging_temperature': 1}

    cat_params = {
        'learning_rate': 0.2,
        'l2_leaf_reg': 3,
        'iterations': 700,
        'depth': 5,
        'border_count': 128,
        'bagging_temperature': 1,
        'random_state': random_state,
        'verbose': 0
    }

    # {'model__kernel': 'rbf', 'model__gamma': 'scale', 'model__epsilon': 0.1, 'model__degree': 4, 'model__C': 100}

    svr_params = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'epsilon': 0.1,
        'degree': 4,
        'C': 100
    }

    #{'model__kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1), 'model__alpha': 1e-05}
    gpr_params = {
        'kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1),
        'alpha': 1e-05
    }

    # Use K-Fold for cross-validation (5-fold by default)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    current_fold = 0

    for train_index, val_index in kfold.split(X, y):
        # Split data

        current_fold += 1
        print(f"Fold {current_fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train, imputer, scaler = process_features(X_train, is_train=True)
        X_val = process_features(X_val, is_train=False, imputer=imputer, scaler=scaler)

        xgboost = xgb.XGBRegressor(**params)
        cbr = CatBoostRegressor(**cat_params)
        svr = SVR(**svr_params)
        gpr = GaussianProcessRegressor(**gpr_params)


        model = StackingRegressor(
            estimators=[
                ('gpr', gpr),
                ('cbr', cbr),
                ('svr', svr)
                ],
            final_estimator=Ridge(),
        )
        # Predict on validation set


        #X_train, indices_to_remove = remove_redundant_features(X_train, correlation_threshold=0.9)
        #X_val = np.delete(X_val, indices_to_remove, axis=1)

        X_train, selector = feature_selection(X_train, y_train)
        X_val = selector.transform(X_val)

        #X_train, indices_to_remove = remove_redundant_features(X_train, correlation_threshold=0.8)
        #X_val = np.delete(X_val, indices_to_remove, axis=1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # Calculate and store metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    # Calculate the mean and std for each metric across folds
    results = {
        'R2 Score': (np.mean(r2_scores), np.std(r2_scores)),
        'Mean Absolute Error': (np.mean(mae_scores), np.std(mae_scores)),
        'Root Mean Squared Error': (np.mean(rmse_scores), np.std(rmse_scores))
    }

    # Print the results
    print("Cross-Validation Performance:")
    for metric, (mean, std) in results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return results

def hyperparameter_tuning_gpr(X, y, n_iter=20, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for a Gaussian Process Regressor (GPR) model using RandomizedSearchCV.

    :param X: Feature matrix
    :param y: Target values
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :param random_state: Random state for reproducibility
    :return: Best model and best parameters
    """
    # Define the parameter grid for Gaussian Process Regressor
    param_grid = {
        'model__alpha': [1e-10, 1e-5, 1e-2, 0.1, 1],  # Noise level
        'model__kernel': [
            C(1.0) * RBF(length_scale=1.0),             # RBF kernel
            C(1.0) * Matern(length_scale=1.0, nu=1.5),  # Matern kernel with smoothness parameter nu
            C(1.0) * RationalQuadratic(length_scale=1.0, alpha=0.5),  # RationalQuadratic kernel
            C(1.0) * DotProduct(sigma_0=1.0)            # DotProduct kernel for linear trends
        ],
    }

    # Initialize the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(random_state=random_state)

    # Build a pipeline to include preprocessing and the model
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector1', VarianceThreshold(0.01)),
        ('selector2', SelectKBest(f_regression, k=100)),
        ('model', gpr)
    ])

    # Set up RandomizedSearchCV with the pipeline
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=3,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Perform the random search
    random_search.fit(X, y)

    # Output the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print(f"Best Cross-Validation R^2 Score:", random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_

def hyperparameter_tuning_xgboost(X, y, n_iter=50, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for an XGBoost model using RandomizedSearchCV.

    :param X: Feature matrix
    :param y: Target values
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :param random_state: Random state for reproducibility
    :return: Best model and best parameters
    """
    # Define the parameter grid for XGBoost
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [100, 200, 300, 500, 700],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0.1, 1, 10, 20, 50],
    }

    # Initialize the XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)

    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector1', VarianceThreshold(0.01)),
        ('selector2', SelectKBest(f_regression, k=100)),
        ('model', xgb_regressor)
    ])

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=3,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Perform the random search
    random_search.fit(X, y)

    # Output the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print(f"Best Cross-Validation R^2 Score:", random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_

def hyperparameter_tuning_catboost(X, y, n_iter=50, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for a CatBoost model using RandomizedSearchCV.

    :param X: Feature matrix
    :param y: Target values
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :param random_state: Random state for reproducibility
    :return: Best model and best parameters
    """
    # Define the parameter grid for CatBoost
    param_grid = {
        'depth': [4, 5, 6, 7, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'iterations': [100, 200, 300, 500, 700],
        'l2_leaf_reg': [1, 3, 5, 7, 10],
        'bagging_temperature': [0, 0.5, 1, 2, 5],
        'border_count': [32, 64, 128, 254],
    }

    # Initialize the CatBoost regressor with early stopping
    catboost_regressor = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=random_state,
        verbose=0,   # Disable output for each iteration
    )

    # Define a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector1', VarianceThreshold(0.01)),
        ('selector2', SelectKBest(f_regression, k=100)),
        ('model', catboost_regressor)
    ])

    # Set up RandomizedSearchCV with the pipeline
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={'model__' + k: v for k, v in param_grid.items()},  # Prefix 'model__' to param grid keys
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=3,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Perform the random search
    random_search.fit(X, y)

    # Output the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print(f"Best Cross-Validation R^2 Score:", random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_


def hyperparameter_tuning_random_forest(X, y, n_iter=50, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for a RandomForestRegressor model using RandomizedSearchCV.

    :param X: Feature matrix
    :param y: Target values
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :param random_state: Random state for reproducibility
    :return: Best model and best parameters
    """
    # Define the parameter grid for RandomForestRegressor
    param_grid = {
        'model__n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
        'model__max_depth': [None, 10, 20, 30, 40, 50],    # Maximum depth of the tree
        'model__min_samples_split': [2, 5, 10, 20],        # Minimum number of samples required to split an internal node
        'model__min_samples_leaf': [1, 2, 4, 8],           # Minimum number of samples required to be at a leaf node
        'model__max_features': ['auto', 'sqrt', 'log2'],    # Number of features to consider when looking for the best split
        'model__bootstrap': [True, False]                   # Whether bootstrap samples are used when building trees
    }

    # Initialize the RandomForestRegressor model
    rf = RandomForestRegressor(random_state=random_state)

    # Build a pipeline to include preprocessing and model
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector1', VarianceThreshold(0.01)),
        ('selector2', SelectKBest(f_regression, k=100)),
        ('model', rf)
    ])

    # Set up RandomizedSearchCV with the pipeline
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=3,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Perform the random search
    random_search.fit(X, y)

    # Output the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print(f"Best Cross-Validation R^2 Score:", random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_

def hyperparameter_tuning_svr(X, y, n_iter=50, cv=5, random_state=42):
    """
    Perform hyperparameter tuning for an SVR model using RandomizedSearchCV.

    :param X: Feature matrix
    :param y: Target values
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :param random_state: Random state for reproducibility
    :return: Best model and best parameters
    """
    # Define the parameter grid for SVR
    param_grid = {
        'model__C': [0.1, 1, 10, 100, 1000],        # Regularization parameter
        'model__epsilon': [0.01, 0.1, 0.2, 0.5, 1],  # Epsilon in the epsilon-SVR model
        'model__kernel': ['linear', 'poly', 'rbf'],  # Kernel type to be used in the algorithm
        'model__degree': [2, 3, 4],                  # Degree of the polynomial kernel function ('poly' only)
        'model__gamma': ['scale', 'auto']            # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    }

    # Initialize the SVR model
    svr = SVR()

    # Build a pipeline to include preprocessing and model
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector1', VarianceThreshold(0.01)),
        ('selector2', SelectKBest(f_regression, k=100)),
        ('model', svr)
    ])

    # Set up RandomizedSearchCV with the pipeline
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=3,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Perform the random search
    random_search.fit(X, y)

    # Output the best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print(f"Best Cross-Validation R^2 Score:", random_search.best_score_)

    return random_search.best_estimator_, random_search.best_params_

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


def local_modeling_approach():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    models, selectors, kmeans = train_cluster_models(X, y, n_clusters=5)

    validate_clustered_model_per_cluster_with_training_score(X, y, n_clusters=5, n_splits=5)

    predictions = make_clustered_predictions(X_test, models, selectors, kmeans)


def xgboost_approach():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    train_evaluate_xgboost(X, y, n_splits=5, random_state=42)

def find_xgboost_hyperparameters():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    #X, imputer, scaler = process_features(X, is_train=True)
    # X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    best_model, best_params = hyperparameter_tuning_xgboost(X, y, n_iter=50, cv=5, random_state=42)

def find_catboost_hyperparameters():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    best_model, best_params = hyperparameter_tuning_catboost(X, y, n_iter=50, cv=5, random_state=42)

def find_svr_hyperparameters():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    best_model, best_params = hyperparameter_tuning_svr(X, y, n_iter=50, cv=5, random_state=42)

def find_hyperparameters_random_forest():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    best_model, best_params = hyperparameter_tuning_random_forest(X, y, n_iter=50, cv=5, random_state=42)

def find_hyperparameters_gpr():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    best_model, best_params = hyperparameter_tuning_gpr(X, y, n_iter=50, cv=5, random_state=42)
def stacked_regressor_approach():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    train_evaluate_stacked_ensemble(X, y, n_splits=10, random_state=42)

def gpr_approach():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    train_evaluate_gpr(X, y, n_splits=10, random_state=42)
def visualize_feature_importance():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    X, selector = feature_selection(X, y)
    X, indices = remove_redundant_features(X)

    #print(len(indices))

    correlation_matrix = pd.DataFrame(X).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def create_submission(sub_name = "y_test_pred"):
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    X, imputer, scaler = process_features(X, is_train=True)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    y_pred = make_prediction_stacked_ensemble(X, y, X_test)

    table = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred.flatten()})

    table.to_csv(f'Data\{sub_name}.csv', index=False)


if __name__ == "__main__":
    #find_catboost_hyperparameters() # {'model__learning_rate': 0.2, 'model__l2_leaf_reg': 3, 'model__iterations': 700, 'model__depth': 5, 'model__border_count': 128, 'model__bagging_temperature': 1}
    #find_xgboost_hyperparameters() #'subsample': 0.7, 'reg_lambda': 1, 'reg_alpha': 10, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
    #find_svr_hyperparameters() # Best Parameters: {'model__kernel': 'rbf', 'model__gamma': 'scale', 'model__epsilon': 0.1, 'model__degree': 4, 'model__C': 100}
    #find_hyperparameters_random_forest() # {'model__n_estimators': 300, 'model__min_samples_split': 2, 'model__min_samples_leaf': 1, 'model__max_features': 'sqrt', 'model__max_depth': 20, 'model__bootstrap': False}
    #find_hyperparameters_gpr() # {'model__kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1), 'model__alpha': 1e-05}
    #xgboost_approach()
    stacked_regressor_approach()
    #gpr_approach()
    #visualize_feature_importance()
    #local_modeling_approach()
    #create_submission(sub_name="submission13_ss")
