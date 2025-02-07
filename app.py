from flask import Flask, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('aviator_model.h5')  # Replace with the correct path to your model

@app.route('/predict', methods=['GET'])
def predict():
    # Example prediction, replace with real logic for prediction
    prediction = model.predict(np.random.rand(1, 10, 1))  # Adjust input shape if necessary
    return jsonify({'prediction': round(float(prediction[0, 0]), 2)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
