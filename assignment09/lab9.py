import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer

# Load dataset
df = pd.read_excel("demo.xlsx")
df = df.dropna(subset=['price'])  # Drop rows where target is missing

# Prepare features and target
X = df.drop(columns=['price', 'id', 'timestamp', 'datetime', 'timezone'], errors='ignore')
y = df['price']

# Identify feature types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define preprocessing steps
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

# Base models and final estimator for stacking
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]
final_estimator = LinearRegression()

# ========== A2: Full pipeline with stacking ==========
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingRegressor(estimators=base_models, final_estimator=final_estimator))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"[Pipeline] Mean Squared Error: {mse:.2f}")

# ========== A3: LIME Explanation (must preprocess manually) ==========

# Manually transform data
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train stacking model separately (outside pipeline)
stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train_transformed, y_train)

# Evaluate again (optional)
manual_preds = stacking_model.predict(X_test_transformed)
manual_mse = mean_squared_error(y_test, manual_preds)
print(f"[Manual] Mean Squared Error: {manual_mse:.2f}")

# LIME Explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_transformed),
    feature_names=preprocessor.get_feature_names_out(),
    mode='regression',
    verbose=True
)

# Explain prediction on one test instance
explanation = explainer.explain_instance(
    data_row=X_test_transformed[0],
    predict_fn=stacking_model.predict
)

# Save explanation to HTML
explanation.save_to_file("lime_explanation.html")
print("LIME explanation saved to lime_explanation.html")
