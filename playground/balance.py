import pandas as pd
from sklearn.utils import resample

# Load data
file_path = "car_kick.csv"
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Ensure the IsBadBuy column is numeric (0s and 1s)
if df["IsBadBuy"].dtype != "int64":
    df["IsBadBuy"] = df["IsBadBuy"].astype(int)

# Separate majority and minority classes
df_majority = df[df.IsBadBuy == 0]
df_minority = df[df.IsBadBuy == 1]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Check the new class counts
print("Balanced dataset:\n", df_balanced.IsBadBuy.value_counts())

# Save the balanced dataset (optional)
df_balanced.to_csv("balanced_car_kick.csv", index=False)

# Print final message
print("Data balancing completed successfully!")
