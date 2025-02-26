from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, 'prediction', 'obesity_dataset.csv')
dataset = pd.read_csv(dataset_path)

# Prepare Data
X = dataset.iloc[:, :-1].values  # Independent variables (BMI, Age, Exercise Hours Per Week)
y = dataset.iloc[:, -1].values   # Target variable (Obesity: Yes/No)

# Encode Target Variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train Model
regressor = LinearRegression()
regressor.fit(X, y)

# View Function for Home Page
def home(request):
    return render(request, 'index.html')

# View Function for Prediction
import numpy as np
from django.shortcuts import render
import joblib  # Assuming you saved the model



def predict(request):
    if request.method == 'POST':
        bmi = float(request.POST['bmi'])
        age = int(request.POST['age'])
        exercise_hours = int(request.POST['exercise_hours'])

        # Prepare Input Data
        input_data = np.array([[bmi, age, exercise_hours]])

        # Make Prediction
        prediction = regressor.predict(input_data)[0]

        # Determine Risk Level
        obesity_risk = 'High' if prediction >= 0.5 else 'Low'

        return render(request, 'index.html', {'prediction': obesity_risk})

    return render(request, 'index.html')
