import os
import pickle
import numpy as np
from django.shortcuts import render
from django.conf import settings

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'prediction', 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

def predict_obesity(request):
    prediction = None
    if request.method == "POST":
        try:
            bmi = float(request.POST.get("bmi"))
            age = float(request.POST.get("age"))
            exercise_hours = float(request.POST.get("exercise_hours"))
            
            # Prepare input features
            input_data = np.array([[bmi, age, exercise_hours]])
            result = model.predict(input_data)[0]
            
            # Convert prediction to readable output
            prediction = "HIGH RISK" if result > 0.5 else "LOW RISK"
        except ValueError:
            prediction = "Invalid input! Please enter numeric values."
    
    return render(request, "index.html", {"prediction": prediction})

def home(request):
    return render(request, 'index.html')