from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get input values
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])
        mean_perimeter = float(request.form['mean_perimeter'])
        mean_area = float(request.form['mean_area'])
        mean_smoothness = float(request.form['mean_smoothness'])
        mean_concavity = float(request.form['mean_concavity'])

        # Create input array
        input_data = np.array([[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_concavity
        ]])

        # Scale data
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]

        # Result
        if prediction == 0:
            result = "Malignant (Cancer Detected)"
            message = "Please consult a medical professional immediately."
            status = "danger"
        else:
            result = "Benign (No Cancer Detected)"
            message = "The tumor appears non-cancerous."
            status = "success"

        return render_template(
            'result.html',
            prediction=result,
            message=message,
            status=status
        )

    except Exception as e:
        return f"Error: {e}"

# Run app
if __name__ == '__main__':
    app.run(debug=True)