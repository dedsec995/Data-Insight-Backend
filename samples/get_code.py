import re

# Your input string
input_string = """
Here's a Python script using pandas for data manipulation and matplotlib with seaborn for creating histogram visualizations based on your requirements. This script assumes that the target column is 'IsBadBuy' (binary classification) and creates histograms for numerical columns along with counts for categorical columns.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = "car_kick.csv"
data = pd.read_csv(file_path)

# Define target column
target_col = 'IsBadBuy'

# List of numerical columns (excluding target)
num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != target_col]

# List of categorical columns (including target)
cat_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]

# Create subplots for numerical columns
fig, axes = plt.subplots(2, len(num_cols)//2, figsize=(15, 7))

for i, col in enumerate(num_cols):
    ax = axes[i//len(num_cols)//2, i%len(num_cols)//2]
    sns.histplot(data=data, x=col, hue=target_col, kde=False, ax=ax)
    ax.set_title(f'{col} Histogram')

# Create a figure for categorical columns (including target)
fig_cat, ax_cat = plt.subplots(1, len(cat_cols), figsize=(15, 5))

for i, col in enumerate(cat_cols):
    ax = ax_cat[i]
    data[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'{col} Counts')

# Adjust subplots
plt.tight_layout()
plt.show()

# Save figures (optional)
fig.savefig('num_histograms.png')
fig_cat.savefig('cat_counts.png')
```
"""

# Regex to find and extract the code block
code_pattern = r'```python(.*?)```'
code_block = re.search(code_pattern, input_string, re.DOTALL)

if code_block:
    code = code_block.group(1).strip()
    print(code)
else:
    print("No code block found.")
