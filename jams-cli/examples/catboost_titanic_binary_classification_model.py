import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
import ssl
import json

# bypass error which causes the script to crash when downloading dataset
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')
target = 'survived'

# preprocessing data

# filling missing value in deck column with a new category: Unknown
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown']
titanic['deck'] = pd.Categorical(
    titanic['deck'], categories=categories, ordered=True)
titanic['deck'] = titanic['deck'].fillna('Unknown')

# filling missing value in age column using mean imputation
age_mean = titanic['age'].fillna(0).mean()
titanic['age'] = titanic['age'].fillna(age_mean)

# droping missing values in embark as there are only 2
titanic = titanic.dropna()

# droping alive column to make the problem more challenging
titanic = titanic.drop('alive', axis=1)

# Create the feature matrix (X) and target vector (y)
X = titanic.drop(target, axis=1)
y = titanic[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# jams-core does not support numeric features as categorical. will be fixed in later releases
X_train['alone'] = X_train['alone'].astype(str)
X_train['adult_male'] = X_train['alone'].astype(str)
X_train['pclass'] = X_train['pclass'].astype(str)
X_train['sibsp'] = X_train['sibsp'].astype(str)
X_train['parch'] = X_train['parch'].astype(str)

# specifying categorical features
categorical_features = ['sex', 'pclass', 'sibsp', 'parch', 'embarked',
                        'class', 'who', 'adult_male', 'embark_town', 'alone', 'deck']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.fit(X_train, y_train)

# predicting accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print accuracy
print(f"Accuracy: {accuracy:.2f}")

# save the model
model.save_model('catboost_titanic', format='cbm')

# ave the sample input
sample_input = X_train.head(10).to_dict(orient='list')
with open("catboost_input.json", "w") as outfile:
    json.dump(sample_input, outfile)
