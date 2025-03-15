import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import DecisionBoundaryDisplay

# Load the dataset
df = pd.read_excel("Lab6_Dataset.xlsx", sheet_name="Sheet1")

# Fill missing price value with median
df.loc[:, 'price'] = df['price'].fillna(df['price'].median())

# Perform equal-width binning into 4 bins
df['price_category'] = pd.cut(df['price'], bins=4, labels=["Low", "Medium", "High", "Very High"])

# Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to calculate Gini Index
def gini_index(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

# Function for Equal Width Binning
def equal_width_binning(data, bins=4):
    return pd.cut(data, bins, labels=range(bins))

# Function for Equal Frequency Binning
def equal_frequency_binning(data, bins=4):
    return pd.qcut(data, bins, labels=range(bins), duplicates='drop')

# Function to determine the best feature using Information Gain
def best_feature_by_information_gain(X, y):
    base_entropy = entropy(y)
    best_info_gain = 0
    best_feature = None
    
    for feature in X.columns:
        values = X[feature].unique()
        feature_entropy = 0
        
        for value in values:
            subset_y = y[X[feature] == value]
            feature_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
        
        info_gain = base_entropy - feature_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    
    return best_feature

# Prepare dataset
df['price_category_encoded'] = df['price_category'].astype('category').cat.codes
X = df[['distance', 'temperature']]  # Keep as DataFrame
y = df['price_category_encoded'].values

# Bin 'distance' column
df['distance_binned'] = equal_width_binning(df['distance'], bins=4)
print("Binned 'distance' column:")
print(df[['distance', 'distance_binned']].head())

# Print entropy and Gini index
print("Entropy of target variable:", entropy(y))
print("Gini Index of target variable:", gini_index(y))

# Identify best feature
best_feature = best_feature_by_information_gain(X, y)
print("Best feature by Information Gain:", best_feature)

# Train Decision Tree
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)

# Plot Decision Tree
plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=['distance', 'temperature'], class_names=['Low', 'Medium', 'High', 'Very High'], filled=True)
plt.show()

# Decision Boundary Visualization
xx, yy = np.meshgrid(np.linspace(df['distance'].min(), df['distance'].max(), 100),
                     np.linspace(df['temperature'].min(), df['temperature'].max(), 100))
xy = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['distance', 'temperature'])

display = DecisionBoundaryDisplay.from_estimator(dt, xy, response_method='predict')
display.ax_.scatter(df['distance'], df['temperature'], c=y, edgecolors='k')
plt.xlabel('Distance')
plt.ylabel('Temperature')
plt.title('Decision Boundary Visualization')
plt.show()
