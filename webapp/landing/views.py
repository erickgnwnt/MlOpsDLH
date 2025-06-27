# import pickle
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import os
# Import the necessary libraries
# from django.views.decorators.csrf import csrf_exempt  # Uncomment if CSRF exemption is needed 



# Path ke model relatif terhadap file views.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'logistic_regression.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = joblib.load(file)

with open(SCALER_PATH, 'rb') as f:
    scaler = joblib.load(f)

def landing(request):
    if request.method == 'POST':
        # Get data from the form
        tss = float(request.POST.get('tss', 0))
        do = float(request.POST.get('do', 0))
        bod = float(request.POST.get('bod', 0))
        cod = float(request.POST.get('cod', 0))
        fosfat = float(request.POST.get('fosfat', 0))
        fecal_coli = float(request.POST.get('fecal_coli', 0))
        total_coliform = float(request.POST.get('total_coliform', 0))

        # Prepare input for prediction as DataFrame with correct columns
        input_data = pd.DataFrame(
            [[tss, do, bod, cod, fosfat, fecal_coli, total_coliform]],
            columns=["TSS", "DO", "BOD", "COD", "Fosfat", "Fecal Coli", "Total-Coliform"]
        )
        # Preprocessing: scaling
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model.predict(input_data_scaled)
        # Get prediction probabilities (confidence)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data_scaled)[0]
            confidence = max(proba)
        else:
            confidence = None

        # Map prediction to status
        status_map = {0: 'Tercemar Ringan', 1: 'Tercemar Sedang', 2: 'Tercemar Berat'}
        water_quality_status = status_map.get(prediction[0], 'Unknown')

        return render(request, 'landing/result.html', {
            'status': water_quality_status,
            'confidence': f"{confidence*100:.2f}%" if confidence is not None else 'N/A'
        })

    return render(request, 'landing/index.html')


