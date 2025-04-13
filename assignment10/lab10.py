import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
import shap
import lime
import lime.lime_tabular
import numpy as np

# Load dataset
df = pd.read_excel("demo.xlsx")  # Replace with your file path if needed
df = df.select_dtypes(include=['number'])  # keep only numeric
df = df.dropna(subset=['price'])

X = df.drop(columns=['price'])
y = df['price']

# --- A1: Correlation Heatmap ---
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap (A1)")
plt.tight_layout()
plt.show()

# --- Standardization ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- A2: PCA with 99% variance ---
pca_99 = PCA(n_components=0.99)
X_pca_99 = pca_99.fit_transform(X_scaled)
X_train_99, X_test_99, y_train_99, y_test_99 = train_test_split(X_pca_99, y, test_size=0.2, random_state=42)
model_99 = LinearRegression().fit(X_train_99, y_train_99)
y_pred_99 = model_99.predict(X_test_99)

print("\n--- A2: PCA (99% Variance) ---")
print("Components retained:", X_pca_99.shape[1])
print("MSE:", mean_squared_error(y_test_99, y_pred_99))
print("R² Score:", r2_score(y_test_99, y_pred_99))

# --- A3: PCA with 95% variance ---
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
X_train_95, X_test_95, y_train_95, y_test_95 = train_test_split(X_pca_95, y, test_size=0.2, random_state=42)
model_95 = LinearRegression().fit(X_train_95, y_train_95)
y_pred_95 = model_95.predict(X_test_95)

print("\n--- A3: PCA (95% Variance) ---")
print("Components retained:", X_pca_95.shape[1])
print("MSE:", mean_squared_error(y_test_95, y_pred_95))
print("R² Score:", r2_score(y_test_95, y_pred_95))

# --- A4: Sequential Feature Selection ---
rf = RandomForestRegressor(random_state=42)
sfs = SequentialFeatureSelector(rf, n_features_to_select=10, direction='forward').fit(X, y)
selected_cols = X.columns[sfs.get_support()]
X_sfs = X[selected_cols]

X_train_sfs, X_test_sfs, y_train_sfs, y_test_sfs = train_test_split(X_sfs, y, test_size=0.2, random_state=42)
rf.fit(X_train_sfs, y_train_sfs)
y_pred_sfs = rf.predict(X_test_sfs)

print("\n--- A4: Sequential Feature Selection ---")
print("Selected Features:", list(selected_cols))
print("MSE:", mean_squared_error(y_test_sfs, y_pred_sfs))
print("R² Score:", r2_score(y_test_sfs, y_pred_sfs))

# --- A5: LIME and SHAP ---
print("\n--- A5: LIME and SHAP Explainability ---")
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sfs.values,
    feature_names=selected_cols.tolist(),
    mode="regression"
)

lime_exp = explainer_lime.explain_instance(
    data_row=X_test_sfs.values[0],
    predict_fn=rf.predict
)
print("\nLIME Explanation (Top features):")
print(lime_exp.as_list())

explainer_shap = shap.TreeExplainer(rf)
shap_values = explainer_shap.shap_values(X_test_sfs)

print("\nSHAP Explanation (Top 10 features by absolute importance):")
shap_importance = np.abs(shap_values).mean(axis=0)
sorted_shap = sorted(zip(selected_cols, shap_importance), key=lambda x: x[1], reverse=True)
for feat, val in sorted_shap[:10]:
    print(f"{feat}: {val:.4f}")

# Optional: SHAP summary plot
shap.summary_plot(shap_values, X_test_sfs, feature_names=selected_cols)
