import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load the dataset
df = pd.read_csv('loan.csv')

# Display the head of the dataframe to understand its structure
print(df.head())

# Check for missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values with the most frequent value or mode (for categorical data) or mean/median (for numerical data)
df['Loan_Default_Risk'].fillna(df['Loan_Default_Risk'].mode()[0], inplace=True)  # Example for Loan_Default_Risk

# Encode categorical variables to numeric values
categorical_cols = ['Marital_Status', 'House_Ownership', 'Vehicle_Ownership', 'Occupation', 'Residence_City', 'Residence_State']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the data into training and testing sets (80-20 split)
X = df.drop('Loan_Default_Risk', axis=1)  # Features
y = df['Loan_Default_Risk']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
if not os.path.exists('model/'):
    os.makedirs('model/')
model_path = 'model/loan_decision_tree_model.pkl'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Save the confusion matrix as a PNG file
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
confusion_matrix_path = 'model/loan_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to {confusion_matrix_path}")

# Generate JSON output with the model and confusion matrix paths, along with the classification report results
report_results = classification_report(y_test, y_pred, output_dict=True)
json_result = {
    'model_path': model_path,
    'conf_path': confusion_matrix_path,
    'result': {
        'accuracy': accuracy,
        'recoil': report_results['weighted avg']['recall'],
        'precision': report_results['weighted avg']['precision'],
        'f1': report_results['weighted avg']['f1-score'],
        'support': report_results['weighted avg']['support']
    }
}

import json
print(json.dumps(json_result, indent=4))