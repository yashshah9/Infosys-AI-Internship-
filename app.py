# Import necessary modules and libraries
from flask import Flask, request, jsonify, render_template, send_from_directory  # Flask web framework components
import numpy as np  # Numerical operations and array manipulation
import base64  # Encode and decode data as base64 strings
from PIL import Image  # Python Imaging Library for image processing
import io  # Core tools for working with streams of data
from tensorflow.keras.models import load_model  # Load pre-trained Keras models

# Create a Flask application instance
app = Flask(__name__)

# Load pre-trained models from HDF5 files
lenet_model = load_model('R:\Infosys-AI-Internship--1\model\lenet_model.h5')  # LeNet model
mlp_model = load_model('R:\Infosys-AI-Internship--1\model\mlp_model.h5')      # MLP model
logistic_model = load_model('R:\Infosys-AI-Internship--1\model\logistic_model.h5')  # Logistic regression model

def preprocess_image(image_data):
    """
    Preprocess the base64 encoded image for model prediction.
    """
    # Decode the base64 image data and extract the image bytes
    image_data = base64.b64decode(image_data.split(',')[1])
    
    # Open the image using PIL and convert to grayscale ('L' mode)
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # Resize the image to 28x28 pixels (MNIST dataset image size)
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Reshape the array to fit the model input shape and normalize
    image = image.reshape(1, 28, 28, 1).astype('float32')
    image = image / 255.0
    
    return image

def predict_with_model(model, image, reshape=False):
    """
    Predict the digit from the image using the specified model.
    """
    # Reshape the image if needed for specific models (MLP and logistic regression)
    if reshape:
        image = image.reshape(1, 28*28)
    
    # Get the prediction probabilities from the model
    prediction = model.predict(image)
    
    # Convert the prediction to a Python list and return
    return prediction.tolist()

# Route to render the main HTML page
@app.route('/')
def index():
    """
    Render the main HTML page.
    """
    return render_template('index.html')

# Route to serve static files (CSS, JavaScript, images) from the 'static' directory
@app.route('/static/<path:path>')
def send_static(path):
    """
    Serve static files.
    """
    return send_from_directory('static', path)

# Route to predict using the LeNet model
@app.route('/predict_lenet', methods=['POST'])
def predict_lenet():
    """
    Predict using the LeNet model.
    """
    # Get the image data from the POST request JSON payload
    image_data = request.json['image_data']
    
    # Preprocess the image for model prediction
    image = preprocess_image(image_data)
    
    # Use the LeNet model to predict the digit
    prediction = predict_with_model(lenet_model, image)
    
    # Return the prediction results as JSON
    return jsonify({'results': prediction[0]})

# Route to predict using the MLP model
@app.route('/predict_mlp', methods=['POST'])
def predict_mlp():
    """
    Predict using the MLP model.
    """
    # Get the image data from the POST request JSON payload
    image_data = request.json['image_data']
    
    # Preprocess the image for model prediction
    image = preprocess_image(image_data)
    
    # Reshape the image for the MLP model
    image = image.reshape(1, 28*28)
    
    # Use the MLP model to predict the digit
    prediction = predict_with_model(mlp_model, image, reshape=True)
    
    # Return the prediction results as JSON
    return jsonify({'results': prediction[0]})

# Route to predict using the logistic regression model
@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    """
    Predict using the logistic regression model.
    """
    # Get the image data from the POST request JSON payload
    image_data = request.json['image_data']
    
    # Preprocess the image for model prediction
    image = preprocess_image(image_data)
    
    # Reshape the image for the logistic regression model
    image = image.reshape(1, 28*28)
    
    # Use the logistic regression model to predict the digit
    prediction = predict_with_model(logistic_model, image, reshape=True)
    
    # Return the prediction results as JSON
    return jsonify({'results': prediction[0]})

# Main entry point: Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
