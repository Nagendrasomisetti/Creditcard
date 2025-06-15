from flask import Flask, render_template, request
import joblib
import numpy as np

# Create app
app = Flask(__name__)

# Load trained model
model = joblib.load('../models/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = [float(x) for x in request.form.values()]
    input_array = np.array(input_data).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    
    return render_template('index.html',
                           prediction=prediction,
                           probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
