from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filePath = "C:\\Users\\HP\\Downloads\\Lab Session Data.xlsx"
thyroidData = pd.read_excel(filePath, sheet_name="thyroid0387_UCI")

numericCols = thyroidData.select_dtypes(include=[np.number]).columns
categoricalCols = thyroidData.select_dtypes(exclude=[np.number]).columns

print(thyroidData.dtypes)
print(thyroidData.describe())

thyroidData[numericCols] = thyroidData[numericCols].apply(
    lambda x: x.fillna(x.median() if ((x - x.mean()).abs() > 3 * x.std()).sum() > 0 else x.mean())
)
thyroidData[categoricalCols] = thyroidData[categoricalCols].apply(lambda x: x.fillna(x.mode()[0]))

print(thyroidData.isnull().sum())

scaler = MinMaxScaler()
thyroidData[numericCols] = scaler.fit_transform(thyroidData[numericCols])

print(thyroidData.head())

vec1 = thyroidData.iloc[0, :].astype(bool)
vec2 = thyroidData.iloc[1, :].astype(bool)

f11 = sum(vec1 & vec2)
f00 = sum(~vec1 & ~vec2)
f01 = sum(~vec1 & vec2)
f10 = sum(vec1 & ~vec2)

JC = f11 / (f01 + f10 + f11)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print("Jaccard Similarity:", JC)
print("Simple Matching Coefficient:", SMC)

vec1Full = thyroidData.loc[0, numericCols].values.astype(np.float64)
vec2Full = thyroidData.loc[1, numericCols].values.astype(np.float64)

cosineSim = np.dot(vec1Full, vec2Full) / (np.linalg.norm(vec1Full) * np.linalg.norm(vec2Full))
print("Cosine Similarity:", cosineSim)

simMatrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        v1 = thyroidData.iloc[i, :].astype(bool)
        v2 = thyroidData.iloc[j, :].astype(bool)
        f11 = sum(v1 & v2)
        f00 = sum(~v1 & ~v2)
        f01 = sum(~v1 & v2)
        f10 = sum(v1 & ~v2)
        JC = f11 / (f01 + f10 + f11)
        SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
        cosineSim = np.dot(thyroidData.loc[i, numericCols].values.astype(np.float64), thyroidData.loc[j, numericCols].values.astype(np.float64)) / \
                    (np.linalg.norm(thyroidData.loc[i, numericCols].values.astype(np.float64)) * np.linalg.norm(thyroidData.loc[j, numericCols].values.astype(np.float64)))
        simMatrix[i, j] = (JC + SMC + cosineSim) / 3

plt.figure(figsize=(12, 8))  
sns.heatmap(simMatrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Similarity'}, 
            xticklabels=range(1, 21), yticklabels=range(1, 21), linewidths=0.5)

plt.title('Similarity Matrix of Thyroid Data', fontsize=16)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Sample Index', fontsize=14)

plt.tight_layout()
plt.show()
