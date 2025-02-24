import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

def load_data(filepath, sheet_name="Sheet1"):
    """Loads data from an Excel file."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def preprocess_data(df, target_column):
    """Selects numerical features, handles missing values, and splits data into independent and target variables."""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_features].dropna()  # Drop rows with missing values
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_linear_regression(X_train, y_train):
    """Trains a linear regression model and returns it."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mape": mean_absolute_percentage_error(y_train, y_train_pred),
        "test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }
    return metrics

def perform_kmeans_clustering(X_train, k_range=(2, 11)):
    """Performs K-Means clustering for different k values and returns evaluation metrics."""
    silhouette_scores, ch_scores, db_indices, distortions = [], [], [], []
    
    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(X_train, kmeans.labels_))
        db_indices.append(davies_bouldin_score(X_train, kmeans.labels_))
    
    return k_range, silhouette_scores, ch_scores, db_indices, distortions

import matplotlib.pyplot as plt

def plot_clustering_metrics(k_values, silhouette_scores, ch_scores, db_indices, distortions):
    """Plots clustering evaluation metrics."""
    
    # Debugging: Print shapes before plotting
    print(f"k_values: {k_values} (len={len(k_values)})")
    print(f"Silhouette Scores: {silhouette_scores} (len={len(silhouette_scores)})")
    print(f"CH Scores: {ch_scores} (len={len(ch_scores)})")
    print(f"DB Indices: {db_indices} (len={len(db_indices)})")
    print(f"Distortions: {distortions} (len={len(distortions)})")
    
    # Find the minimum valid length across all lists
    min_length = min(len(k_values), len(silhouette_scores), len(ch_scores), len(db_indices), len(distortions))

    # Trim all lists to ensure they are of equal length
    k_values = k_values[:min_length]
    silhouette_scores = silhouette_scores[:min_length]
    ch_scores = ch_scores[:min_length]
    db_indices = db_indices[:min_length]
    distortions = distortions[:min_length]

    # Ensure there's enough data to plot
    if min_length < 2:
        print("Error: Not enough valid data points for plotting. Ensure k_values and metric lists are correctly populated.")
        return

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(k_values, silhouette_scores, marker='o', label="Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(k_values, ch_scores, marker='o', label="Calinski-Harabasz Score", color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("CH Score")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(k_values, db_indices, marker='o', label="Davies-Bouldin Index", color='r')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("DB Index")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(k_values, distortions, marker='o', label="Distortion", color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Load and preprocess data
    filepath = r"User/Pavan/Downloads/assignment05/demo"
    df = load_data(filepath)
    X, y = preprocess_data(df, target_column='temperatureMin')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate linear regression model
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Print model evaluation results
    print("Coefficients:", metrics["coefficients"])
    print("Intercept:", metrics["intercept"])
    print("Train MSE:", metrics["train_mse"], "Test MSE:", metrics["test_mse"])
    print("Train RMSE:", metrics["train_rmse"], "Test RMSE:", metrics["test_rmse"])
    print("Train MAPE:", metrics["train_mape"], "Test MAPE:", metrics["test_mape"])
    print("Train R2 Score:", metrics["train_r2"], "Test R2 Score:", metrics["test_r2"]) 
    
    # Perform K-Means clustering
    k_values, silhouette_scores, ch_scores, db_indices, distortions = perform_kmeans_clustering(X_train, k_range=(2, 20))
    
    # Plot clustering metrics
    plot_clustering_metrics(k_values, silhouette_scores, ch_scores, db_indices, distortions)