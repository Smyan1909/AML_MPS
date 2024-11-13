from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor, RandomForestRegressor, IsolationForest
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
        ('univariate', SelectKBest(f_regression, k=205))
    ])
    
    X_selected = selector.fit_transform(X, y)

    return X_selected, selector


def load_data():
    """
    Load the data set

    :return: data set
    """
    X_train = pd.read_csv('X_train.csv').drop(columns=['id']).to_numpy()
    y_train = pd.read_csv('y_train.csv').drop(columns=['id']).to_numpy().ravel()
    X_test = pd.read_csv('X_test.csv').drop(columns=['id']).to_numpy()


    return X_train, y_train, X_test

def detect_outliers_IsolationForest(standardized_features, n_estimators = 100, contamination=0.025):
    """
    Detect outliers in the dataset using Isolation Forest.
    :param standardized_features: The standardized features of the dataset.
    :param contamination: The amount of contamination of the data set (default is 0.01).
    :return: A boolean array indicating which samples are outliers.
    """

    scaler = RobustScaler()

    standardized_features = scaler.fit_transform(standardized_features)

    imputer = SimpleImputer(strategy="median")

    standardized_features = imputer.fit_transform(standardized_features)

    pca = PCA(n_components=2)
    standardized_features = pca.fit_transform(standardized_features)

    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso_forest.fit(standardized_features)

    predictions = iso_forest.predict(standardized_features)

    anomalies = predictions <= 0

    return anomalies


def remove_outliers(X, y=None):
    """
    Remove outliers from the data set

    :param data: data set
    :return: data set without outliers
    """
    outliers = detect_outliers_IsolationForest(X, contamination=0.045) # 0.075
    #print("Number of True values:", sum(outliers))

    if y is not None:
        return X[~outliers], y[~outliers], outliers
    else:
        return X[~outliers], outliers
    
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
        #print(X_scaled.shape)

        return X_scaled, imputer, scaler
    else:
        # Use the fitted imputer and scaler for test data
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        return X_scaled

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

def train_evaluate_stacked_ensemble(X, y, n_splits=5, random_state=47):
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
        'kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1.0),
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
        #print("hi")

        #X_train, indices_to_remove = remove_redundant_features(X_train, correlation_threshold=0.8)
        #X_val = np.delete(X_val, indices_to_remove, axis=1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # Calculate and store metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        #print("shy")

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

def make_prediction_stacked_ensemble(X, y, X_test, random_state=47):
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
        'kernel': 1**2 * RationalQuadratic(alpha=0.5, length_scale=1.0),
        'alpha': 1e-05
    }

    X, selector = feature_selection(X, y)
    #print(X.shape)
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



def stacked_regressor_approach():
    X_train, y_train, X_test = load_data()

    X, y, outliers = remove_outliers(X_train, y_train)

    train_evaluate_stacked_ensemble(X, y, n_splits=10, random_state=1)


def create_submission(sub_name = "y_test_pred"):
    X_train, y_train, X_test = load_data()
    #print(X_train.shape)

    X, y, outliers = remove_outliers(X_train, y_train)
    #print(X.shape)

    X, imputer, scaler = process_features(X, is_train=True)
    #print(X.shape)
    X_test = process_features(X_test, is_train=False, imputer=imputer, scaler=scaler)

    y_pred = make_prediction_stacked_ensemble(X, y, X_test)

    table = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred.flatten()})

    table.to_csv(f'Data/{sub_name}.csv', index=False)



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
    create_submission(sub_name="submission_23")