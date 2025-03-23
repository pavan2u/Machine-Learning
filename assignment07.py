import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

# Load the dataset
df = pd.read_excel(r"C:\Users\Pavan Sankar\Downloads\demo.xlsx", sheet_name="Sheet1")

# Fill missing price value with median
df.loc[:, 'price'] = df['price'].fillna(df['price'].median())

# Perform equal-width binning into 4 bins
df['price_category'] = pd.cut(df['price'], bins=4, labels=["Low", "Medium", "High", "Very High"])

# Prepare dataset for classification
df['price_category_encoded'] = df['price_category'].astype('category').cat.codes
X = df[['distance', 'temperature']]
y = df['price_category_encoded'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=500, random_state=42)
}

# Train and evaluate classifiers
classification_results = []
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    classification_results.append([name, train_acc, test_acc, precision, recall, f1])

# Convert classification results into a DataFrame and display
classification_results_df = pd.DataFrame(classification_results, columns=["Classifier", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"])
print("Classification Results:")
print(classification_results_df)

# Define regressors
regressors = {
    "Decision Tree Regressor": GradientBoostingRegressor(random_state=42),
    "Random Forest Regressor": RandomForestClassifier(random_state=42),
    "AdaBoost Regressor": AdaBoostClassifier(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42),
    "CatBoost Regressor": CatBoostRegressor(verbose=0, random_state=42),
    "SVR": SVR(),
    "MLP Regressor": MLPRegressor(max_iter=500, random_state=42)
}

# Train and evaluate regressors
regression_results = []
for name, model in regressors.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    regression_results.append([name, train_mae, test_mae, train_mse, test_mse, train_r2, test_r2])

# Convert regression results into a DataFrame and display
regression_results_df = pd.DataFrame(regression_results, columns=["Regressor", "Train MAE", "Test MAE", "Train MSE", "Test MSE", "Train R2", "Test R2"])
print("Regression Results:")
print(regression_results_df)

# Apply clustering algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clustering_algorithms = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=4)
}

# Train clustering models
clustering_results = {}
for name, model in clustering_algorithms.items():
    clustering_results[name] = model.fit_predict(X_scaled)
    df[f"Cluster_{name}"] = clustering_results[name]

# Display cluster assignments
print("Clustering Results:")
for name in clustering_algorithms.keys():
    print(f"{name} Cluster Assignments:")
    print(df[["distance", "temperature", f"Cluster_{name}"]].head())

# Plot Decision Tree
plt.figure(figsize=(10,6))
plot_tree(classifiers["Decision Tree"], feature_names=['distance', 'temperature'], class_names=['Low', 'Medium', 'High', 'Very High'], filled=True)
plt.show()
