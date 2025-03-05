from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model and preprocessing artifacts
with open("columns.pkl", 'rb') as f:
    all_features_breast_cancer = pickle.load(f)
with open("scaler.pkl", 'rb') as f:
    scalers_breast_cancer = pickle.load(f)
with open("parkinsons_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_breast_cancer = pickle.load(f)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/prevention', methods=['GET'])
def prevention_page():
    return render_template('prevention.html')

@app.route('/causes', methods=['GET'])
def causes_page():
    return render_template('causes.html')

@app.route('/hospitals', methods=['GET'])
def hospitals_page():
    return render_template('hospitals.html')

@app.route('/research', methods=['GET'])
def research_page():
    return render_template('research.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form data
        input_data = [
            float(request.form.get("mean_radius", 0)),
            float(request.form.get("mean_texture", 0)),
            float(request.form.get("mean_perimeter", 0)),
            float(request.form.get("mean_area", 0))
        ]

        # Convert input data to numpy array for processing
        input_array = np.array([input_data])

        # Apply custom logic before prediction
        if np.all(input_array < 0):
            prediction_index = 2  # Custom index for "Uncertain"
        else:
            # Proceed with normal prediction
            input_array = scalers_breast_cancer.transform(input_array)
            prediction_index = loaded_model_xgb_breast_cancer.predict(input_array)[0]

        # Translate index into a meaningful result
        if prediction_index == 0:
            result = "Malignant"
            result_color = "red"
            suggestions = "Consult an oncologist immediately."
            recommended_hospitals = ["Specialist Hospital 1", "Oncology Center 2"]
        elif prediction_index == 1:
            result = "Benign"
            result_color = "green"
            suggestions = "Maintain regular check-ups and a healthy lifestyle."
            recommended_hospitals = ["Community Clinic A", "General Hospital B"]
        else:
            result = "Uncertain"
            result_color = "orange"
            suggestions = "Further tests are required to confirm the diagnosis."
            recommended_hospitals = ["Specialist Diagnostic Center C", "Advanced Diagnostic Lab D"]

        # Render the result on the predict page
        return render_template('predict.html', 
                               prediction=result, 
                               result_color=result_color, 
                               suggestions=suggestions, 
                               recommended_hospitals=recommended_hospitals)

    except Exception as e:
        return render_template('predict.html', 
                               prediction="An error occurred.", 
                               result_color="gray", 
                               suggestions="Please check the inputs and try again.", 
                               recommended_hospitals=[])

if __name__ == "__main__":
    app.run(debug=True)
