import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import warnings

def generate_classification_data(n_samples=1000, n_features=10, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    return confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred), train_report, test_report

def plot_confusion_matrices(train_matrix, test_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(train_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Training Confusion Matrix")
    sns.heatmap(test_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title("Testing Confusion Matrix")
    plt.show()

def load_stock_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def preprocess_stock_data(df):
    df = df.dropna()

    # Convert 'M' (millions) and 'K' (thousands) in Volume column
    def convert_numeric(value):
        if isinstance(value, str):
            if 'M' in value:
                return float(value.replace('M', '')) * 1e6
            elif 'K' in value:
                return float(value.replace('K', '')) * 1e3
        return float(value)

    df['Volume'] = df['Volume'].apply(convert_numeric)  # Apply conversion to Volume column

    X = df[['Open', 'High', 'Low', 'Volume']].values  # Features
    y = df['Price'].values  # Target variable

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def generate_knn_data():
    X = np.random.uniform(1, 10, 20)
    Y = np.random.uniform(1, 10, 20)
    labels = (X + Y > 10).astype(int)
    return np.column_stack((X, Y)), labels

def train_knn_classifier(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def generate_test_grid():
    grid_x, grid_y = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))

def plot_knn_results(X_test, y_pred, title):
    colors = ['blue' if c == 0 else 'red' for c in y_pred]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, edgecolors='black', s=10, alpha=0.5)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(title)
    plt.grid(True)
    plt.show()

def tune_knn_hyperparameters(X_train, y_train):
    param_dist = {'n_neighbors': np.arange(1, 20, 2)}
    knn = KNeighborsClassifier()
    search = RandomizedSearchCV(knn, param_dist, cv=5, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_params_['n_neighbors']


def main():
    # Random Forest Classification
    X_train, X_test, y_train, y_test = generate_classification_data()
    model = train_random_forest(X_train, y_train)
    conf_train, conf_test, train_report, test_report = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
    plot_confusion_matrices(conf_train, conf_test)
    print("Training Classification Report:")
    print(pd.DataFrame(train_report).transpose())
    print("\nTesting Classification Report:")
    print(pd.DataFrame(test_report).transpose())
    
    file_path = r"User/Pavan/Downloads/assignment04/Lab_Session_Data"
    sheet_name = "IRCTC Stock Price"
    df = load_stock_data(file_path, sheet_name)
    X_train, X_test, y_train, y_test = preprocess_stock_data(df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    regression_model = train_linear_regression(X_train, y_train)
    train_metrics = calculate_regression_metrics(y_train, regression_model.predict(X_train))
    test_metrics = calculate_regression_metrics(y_test, regression_model.predict(X_test))
    
    # kNN Classification
    warnings.filterwarnings("ignore")
    X_train_knn, y_train_knn = generate_knn_data()
    test_data = generate_test_grid()
    best_k = tune_knn_hyperparameters(X_train_knn, y_train_knn)
    knn_model = train_knn_classifier(X_train_knn, y_train_knn, k=best_k)
    y_pred_knn = knn_model.predict(test_data)
    plot_knn_results(test_data, y_pred_knn, f"kNN Classification with Best k={best_k}")
    print(f"Best k found: {best_k}")

if __name__ == "__main__":
    main()