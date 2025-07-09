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
MODEL_PATHS = {
    "Logistic Regression": os.path.join(PROJECT_ROOT, 'models', 'logistic_regression.pkl'),
    "Random Forest": os.path.join(PROJECT_ROOT, 'models', 'random_forest.pkl'),
    "KNN": os.path.join(PROJECT_ROOT, 'models', 'knn.pkl'),
    "Ordinal Logistic Regression": os.path.join(PROJECT_ROOT, 'models', 'ordinal_logistic_regression.pkl')
}
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')

models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

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
        # Prediksi dengan semua model
        status_map = {0: 'Tercemar Ringan', 1: 'Tercemar Sedang', 2: 'Tercemar Berat'}
        results = {}
        for name, model in models.items():
            pred = model.predict(input_data_scaled)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data_scaled)[0]
                confidence = max(proba)
            else:
                confidence = None
            results[name] = {
                'status': status_map.get(pred[0], 'Unknown'),
                'confidence': f"{confidence*100:.2f}%" if confidence is not None else 'N/A'
            }
        # Feature importance hanya untuk Random Forest
        feature_importance = None
        if "Random Forest" in models:
            rf_model = models["Random Forest"]
            if hasattr(rf_model, "feature_importances_"):
                feature_importance = list(zip(
                    ["TSS", "DO", "BOD", "COD", "Fosfat", "Fecal Coli", "Total-Coliform"],
                    rf_model.feature_importances_
                ))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
        # Koefisien untuk Logistic Regression
        logreg_coeff = None
        if "Logistic Regression" in models:
            logreg_model = models["Logistic Regression"]
            if hasattr(logreg_model, "coef_"):
                logreg_coeff = list(zip(
                    ["TSS", "DO", "BOD", "COD", "Fosfat", "Fecal Coli", "Total-Coliform"],
                    logreg_model.coef_[0]
                ))
                logreg_coeff.sort(key=lambda x: abs(x[1]), reverse=True)
        return render(request, 'landing/result.html', {
            'results': results,
            'feature_importance': feature_importance,
            'logreg_coeff': logreg_coeff
        })

    return render(request, 'landing/index.html')


