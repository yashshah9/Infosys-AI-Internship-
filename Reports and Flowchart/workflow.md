# Workflow Overview
## Frontend Interface (index.html)

## HTML Structure:

The HTML file sets up the structure and layout of the web page using Tailwind CSS for styling.
Key elements include:
A header section with the title and description.
A main section with a canvas for drawing digits and a dropdown to select different models.
A section for displaying prediction results and a bar chart.
Buttons for clearing the canvas and making predictions.
Frontend Functionality (scripts.js)

## Canvas Drawing:

- **Initialization**: The init() function initializes the canvas and sets up event listeners for mouse and touch events.
- **Drawing**: Functions like sketchpad_mouseDown(), sketchpad_mouseMove(), sketchpad_mouseUp(),    sketchpad_touchStart(), and sketchpad_touchMove() handle the drawing on the canvas.
- **Clear Canvas**: The clear_button event listener clears the canvas when clicked.
- **Prediction**:
    The predict() function sends the drawn digit to the backend for prediction based on the selected model.
    The predict_all() function sends the digit to all models for prediction.
- **Displaying Results**:
    The results from the backend are displayed in the prediction result section.
    The updateChart() function updates the bar chart with the confidence percentages from the predictions.
- **Backend Server (app.py)**

## Flask Application:

The Flask application serves the frontend and handles API requests.
Routes like '/predict_lenet', '/predict_mlp', and '/predict_logistic' receive the image data from the frontend, process it, and return the prediction results.

### Model Prediction:
 - Each model endpoint processes the image, feeds it into the respective trained model, and returns the prediction probabilities.

## Detailed Workflow:

**User Interaction:**

- The user draws a digit on the canvas using the mouse or touch input.
- The user selects a model from the dropdown and clicks the "Predict" button.
  Canvas Image Data:
- The canvas drawing is converted to image data (base64 URL) in the predict() function.

**Backend Request:**

- The fetch() API sends this image data to the Flask backend at the appropriate model endpoint (e.g., '/predict_lenet').

**Model Prediction:**

The backend processes the image, makes a prediction using the selected model, and sends the results back as a JSON response.

**Display Results:**

- The frontend receives the prediction results.
- The predicted digit and confidence percentages are displayed, and the bar chart is updated to reflect the model's  confidence in each digit class.

**Code Integration**

 - index.html: Provides the user interface, with elements for drawing, selecting models, and displaying results.
 - scripts.js: Implements the logic for drawing on the canvas, handling user interactions, sending image data to the backend, and updating the UI with the results.
 - app.py: Handles the server-side logic, processing incoming image data, running predictions using pre-trained models, and returning the results to the frontend.