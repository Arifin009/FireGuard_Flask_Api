from flask import Flask, jsonify, request
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request form and convert to float
        temp = float(request.form.get('temp'))
        hum = float(request.form.get('hum'))
        gas = float(request.form.get('gas'))
        co = float(request.form.get('co'))

        # Create input query as numpy array
        input_query = np.array([[temp, hum, gas, co]])

        # Scale the input values
        input_query_scaled = scaler.transform(input_query)

        # Make prediction using the model
        result = model.predict(input_query_scaled)[0]

        # # Map the prediction to air quality labels
        # air_quality_map = {0: 'Poor', 1: 'Moderate', 2: 'Good'}
        # predicted_quality = air_quality_map[result]

        # Return prediction result as JSON
        return jsonify({
            'prediction': str(result)
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

# if __name__ == '__main__':
#     app.run(debug=True)
