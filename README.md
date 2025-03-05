# Breast Cancer Detection using Machine Learning

## Overview
This project is a web-based application for predicting breast cancer based on input features using a machine learning model. It includes a Flask-based backend that processes user input, makes predictions using a trained model, and provides recommendations based on the results.

## Features
- User-friendly web interface with Bootstrap styling.
- Predicts breast cancer as **Benign, Malignant, or Uncertain**.
- Provides health recommendations and hospital suggestions.
- Uses a pre-trained **XGBoost model** for classification.
- Flask-based backend with endpoints for different pages.

## Technologies Used
- **Python** (Flask, NumPy, Pandas, Scikit-learn, XGBoost)
- **HTML, CSS, Bootstrap** (for frontend UI)
- **Pickle** (for model serialization)
- **Jupyter Notebook** (for model training and evaluation)

## Project Structure
```
Breast-Cancer-Detection/
│── templates/                # HTML files (frontend pages)
│   ├── index.html
│   ├── predict.html
│   ├── prevention.html
│   ├── causes.html
│   ├── hospitals.html
│   ├── research.html
│── static/                   # CSS & JS files
│   ├── styles.css
│── models/                   # Machine Learning Model
│   ├── breast_cancer_model.pkl
│   ├── scaler.pkl
│── app.py                     # Flask Application
│── requirements.txt            # Dependencies
│── README.md                   # Project Documentation
```

## Installation & Setup
### Prerequisites:
- Python 3.8+
- Virtual environment (optional but recommended)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/breast-cancer-detection.git
   cd breast-cancer-detection
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open a browser and go to `http://127.0.0.1:5000/` to access the app.

## Usage
1. **Navigate to the Prediction Page** (`/predict`)
2. **Enter feature values** in the form.
3. **Click the 'Predict' button** to get results.
4. **View results** including prediction, recommendations, and hospital suggestions.

## Model Training
The model was trained using the **Breast Cancer Wisconsin Dataset**. The preprocessing steps included:
- Handling missing values
- Feature scaling using `StandardScaler`
- Model training with **XGBoost** classifier
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Final model saved using `pickle`

## Endpoints
| Route | Method | Description |
|--------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | GET | Render prediction form |
| `/predict` | POST | Process prediction |
| `/prevention` | GET | Prevention info page |
| `/causes` | GET | Causes info page |
| `/hospitals` | GET | Hospitals list |
| `/research` | GET | Research papers page |

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Description"`)
4. Push to the branch (`git push origin feature-name`)
5. Create a Pull Request

## License
This project is licensed under the MIT License.

## Contact
For any queries or contributions, contact **[Sravani Parvathaneni]** at **sravani.m@makeskilled.com**

