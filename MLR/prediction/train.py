# Importing Libraries
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Importing Dataset
dataset = pd.read_csv(r'D:\MLR\MLR\prediction\obesity_dataset.csv')

# Splitting features and target variable
X = dataset.iloc[:, :-1].values  # Independent variables (BMI, Age, Exercise Hours Per Week)
y = dataset.iloc[:, -1].values   # Target variable (Obesity: Yes/No)

# Encoding the Target Variable (Yes -> 1, No -> 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting Dataset into Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Multiple Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set
y_pred = regressor.predict(X_test)

# Checking Accuracy
accuracy = regressor.score(X_test, y_test)
print("Model Accuracy:", accuracy)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(regressor, f)

print(f"✅ Model trained and saved successfully at {model_path}")