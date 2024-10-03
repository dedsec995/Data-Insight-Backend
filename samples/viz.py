import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load data
file_path = "car_kick.csv"
data = pd.read_csv(file_path)

# Select numeric columns excluding the target variable
numeric_data = data.select_dtypes(include=np.number).drop('IsBadBuy', axis=1)

# Calculate correlation matrix
corr_matrix = numeric_data.corr()

# Save correlation matrix as JSON
with open('corr.json', 'w') as f:
    json.dump(corr_matrix.to_dict(), f, indent=4)

# Visualize correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    linewidths=.5,
    cbar=False,
    square=True,
    cmap="YlGnBu"
)
plt.title("Correlation Heatmap of Numeric Values")
plt.savefig('corr.png')
plt.show()