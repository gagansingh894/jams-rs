import lightgbm as lgb
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ssl
import json

# bypass error which causes the script to crash when downloading dataset
ssl._create_default_https_context = ssl._create_unverified_context


# Step 1: Load the Iris dataset
iris = sns.load_dataset('iris')
# Display the first few rows to understand the data structure
print(iris.head())

# Step 2: Preprocess the data
# Convert the dataset to a binary classification problem
# For this, we'll select only 'setosa' and 'versicolor' classes
iris = iris[iris['species'] != 'virginica']

# Map 'setosa' to 0 and 'versicolor' to 1
iris['species'] = iris['species'].map({'setosa': 0, 'versicolor': 1})

# Separate features and target variable
X = iris.drop(['species'], axis=1)
y = iris['species']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define parameters for LightGBM
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',  # Evaluation metric
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Step 4: Train the model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Make predictions
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_class = np.round(y_pred)  # Convert probabilities to binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

# Step 9: Save model and sample output
bst.save_model("lightgbm_iris.txt")
sample_input = X_train.head(10).to_dict(orient='list')
with open("lightgbm_input.json", "w") as outfile:
    json.dump(sample_input, outfile)
