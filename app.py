from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained models with error handling (no need for label_encoder.pkl here)
try:
    stacking_model = joblib.load('C:\\Users\\hp\\Desktop\\data science\\stacking_classifier_model.pkl')
    xgboost_model = joblib.load('C:\\Users\\hp\\Desktop\\data science\\xgboost_model.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    stacking_model = None
    xgboost_model = None

# Define feature names based on your dataset
feature_names = ['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 'MCV', 
                 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']

# Explicit mapping of numeric predictions to diagnosis labels
diagnosis_mapping = {
    0: "Healthy",
    1: "Iron deficiency anemia",
    2: "Leukemia",
    3: "Leukemia with thrombocytopenia",
    4: "Normocytic normochromic anemia"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if models loaded successfully
    if stacking_model is None or xgboost_model is None:
        return render_template('index.html', 
                             error="Failed to load models. Check server logs.")

    try:
        # Get form data
        data = [float(request.form[feature]) for feature in feature_names]
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Make predictions (returns numeric labels: 0, 1, 2, 3, 4)
        stacking_pred = stacking_model.predict(input_data)[0]
        xgboost_pred = xgboost_model.predict(input_data)[0]
        
        # Map predictions to diagnosis labels
        stacking_result = diagnosis_mapping.get(stacking_pred, "Unknown")
        xgboost_result = diagnosis_mapping.get(xgboost_pred, "Unknown")
        
        # Render the results
        return render_template('index.html', 
                             stacking_prediction=stacking_result, 
                             xgboost_prediction=xgboost_result)
    
    except Exception as e:
        # Handle any errors during prediction (e.g., invalid input)
        return render_template('index.html', 
                             error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)