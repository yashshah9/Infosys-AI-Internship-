from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import base64
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load pre-trained models
lenet_model = load_model('R:\Infosys-AI-Internship--1\model\lenet_model.h5')
mlp_model = load_model('R:\Infosys-AI-Internship--1\model\mlp_model.h5')
logistic_model = load_model('R:\Infosys-AI-Internship--1\model\logistic_model.h5')

def preprocess_image(image_data):
    """
    Preprocess the base64 encoded image for model prediction.
    """
    # Decode the base64 image data
    image_data = base64.b64decode(image_data.split(',')[1])
    # Open the image and convert to grayscale
    image = Image.open(io.BytesIO(image_data)).convert('L')
    # Resize image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert image to numpy array
    image = np.array(image)
    # Reshape array to fit the model input and normalize
    image = image.reshape(1, 28, 28, 1).astype('float32')
    image = image / 255.0
    return image

def predict_with_model(model, image, reshape=False):
    """
    Predict the digit from the image using the specified model.
    """
    # Reshape the image if needed for specific models
    if reshape:
        image = image.reshape(1, 28*28)  # Reshape for MLP and logistic regression
    # Get the prediction probabilities
    prediction = model.predict(image)
    return prediction.tolist()  # Return the prediction probabilities as a list

@app.route('/')
def index():
    """
    Render the main HTML page.
    """
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """
    Serve static files.
    """
    return send_from_directory('static', path)

@app.route('/predict_lenet', methods=['POST'])
def predict_lenet():
    """
    Predict using the LeNet model.
    """
    image_data = request.json['image_data']
    image = preprocess_image(image_data)
    prediction = predict_with_model(lenet_model, image)
    return jsonify({'results': prediction[0]})  # Return the prediction probabilities

@app.route('/predict_mlp', methods=['POST'])
def predict_mlp():
    """
    Predict using the MLP model.
    """
    image_data = request.json['image_data']
    image = preprocess_image(image_data)
    image = image.reshape(1, 28*28)  # Reshape for MLP
    prediction = predict_with_model(mlp_model, image, reshape=True)
    return jsonify({'results': prediction[0]})  # Return the prediction probabilities

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    """
    Predict using the logistic regression model.
    """
    image_data = request.json['image_data']
    image = preprocess_image(image_data)
    image = image.reshape(1, 28*28)  # Reshape for logistic regression
    prediction = predict_with_model(logistic_model, image, reshape=True)
    return jsonify({'results': prediction[0]})  # Return the prediction probabilities

if __name__ == '__main__':
    app.run(debug=True)
