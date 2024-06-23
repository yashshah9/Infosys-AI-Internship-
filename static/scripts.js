// Global variables for canvas and drawing context
var canvas, ctx;
var mouseX, mouseY, mouseDown = 0;
var lastX, lastY;
var touchX, touchY;
var barChart;
var confidencePercentages = Array(10).fill(0); // Array to store confidence percentages initialized to zero

// Global Variables: These variables are declared globally to be accessible throughout the script.
// canvas and ctx: These will store the canvas element and its 2D drawing context, respectively.
// mouseX, mouseY: Hold current mouse coordinates on the canvas.
// mouseDown: Indicates whether the mouse button is currently pressed (0 for false, 1 for true).
// lastX, lastY: Store the last recorded coordinates of the mouse.
// touchX, touchY: Hold current touch coordinates on the canvas (for touch devices).
// barChart: Stores the instance of the chart object.
// confidencePercentages: An array initialized with 10 zeros, intended to store confidence percentages for predictions.





//-----------------------------------------------------------------------------------------------------------------//





function init() {
    canvas = document.getElementById('sketchpad'); // Get canvas element
    ctx = canvas.getContext('2d'); // Get 2D drawing context
    ctx.fillStyle = "black"; // Set initial background color
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill canvas with black color
 
    if (ctx) 
    {
         // Event listeners for mouse events
        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
        window.addEventListener('mouseup', sketchpad_mouseUp, false);

         // Event listeners for touch events
        canvas.addEventListener('touchstart', sketchpad_touchStart, false);
        canvas.addEventListener('touchmove', sketchpad_touchMove, false);
    }
}
// init() Function: This function initializes the canvas and sets up event listeners for drawing and touch events.
// Retrieves the canvas element and its 2D context (ctx).
// Sets the initial background color of the canvas to black.
// Adds event listeners for mouse and touch events (mousedown, mousemove, mouseup, touchstart, touchmove), calling respective handler functions (sketchpad_mouseDown, sketchpad_mouseMove, etc.).





//-----------------------------------------------------------------------------------------------------------------//





// Function to draw on the canvas
function draw(ctx, x, y, size, isDown) 
{
    if (isDown) 
    {
        ctx.beginPath();
        ctx.strokeStyle = "white"; // Set stroke color to white
        ctx.lineWidth = '15'; // Set line width
        ctx.lineJoin = ctx.lineCap = 'round'; // Set line join and cap styles to round
        ctx.moveTo(lastX, lastY); // Move to the last recorded position
        ctx.lineTo(x, y); // Draw a line to the current position
        ctx.closePath();
        ctx.stroke(); // Perform the drawing
    }
    lastX = x;
    lastY = y;
}
// draw() Function: This function is responsible for drawing on the canvas.
// It takes parameters for context (ctx), coordinates (x, y), size of the drawing tool (size), and whether the mouse is currently down (isDown).
// If isDown is true, it begins a new path, sets drawing properties (color, line width, join style), draws a line from the last position to the current one, and strokes it.
// Updates lastX and lastY to the current coordinates.





//-----------------------------------------------------------------------------------------------------------------//





// Mouse down event handler
function sketchpad_mouseDown(e) 
{
    mouseDown = 1;
    getMousePos(e);
    draw(ctx, mouseX, mouseY, 12, false); // Start drawing
}
// sketchpad_mouseDown(e): Event handler for mouse down.
// Sets mouseDown to 1 to indicate the mouse button is pressed.
// Retrieves mouse coordinates using getMousePos(e).
// Calls draw() to start drawing on the canvas.





// Mouse up event handler
function sketchpad_mouseUp() 
{ 
    mouseDown = 0; // Stop drawing
}
// sketchpad_mouseUp(): Event handler for mouse up.
// Sets mouseDown to 0, indicating the mouse button is released and stops drawing.





// Mouse move event handler
function sketchpad_mouseMove(e) 
{
    getMousePos(e);
    if (mouseDown == 1) {
        draw(ctx, mouseX, mouseY, 12, true); // Continue drawing

    }
}
// sketchpad_mouseMove(e): Event handler for mouse move.
// Updates mouse coordinates using getMousePos(e).
// If mouseDown is true, calls draw() to continue drawing on the canvas.





// Function to get mouse position relative to the canvas
function getMousePos(e) 
{
    mouseX = e.offsetX ? e.offsetX : e.layerX;
    mouseY = e.offsetY ? e.offsetY : e.layerY;
}
// getMousePos(e): Helper function to get mouse position relative to the canvas.
// Determines mouse coordinates using e.offsetX and e.offsetY (standard properties) or fallbacks (e.layerX and e.layerY for compatibility).





//-----------------------------------------------------------------------------------------------------------------//






// Touch start event handler
function sketchpad_touchStart(e) 
{
    getTouchPos(e);
    draw(ctx, touchX, touchY, 12, false); // Start drawing

    e.preventDefault();
}
// sketchpad_touchStart(e): Event handler for touch start.
// Gets touch coordinates using getTouchPos(e).
// Calls draw() to start drawing on the canvas.
// Prevents default touch behavior to avoid scrolling or other unintended actions.





// Touch move event handler
function sketchpad_touchMove(e) 
{
    getTouchPos(e);
    draw(ctx, touchX, touchY, 12, true); // Continue drawing
    e.preventDefault();
}
// sketchpad_touchMove(e): Event handler for touch move.
// Gets touch coordinates using getTouchPos(e).
// Calls draw() to continue drawing on the canvas.
// Prevents default touch behavior to avoid scrolling.





// Function to get touch position relative to the canvas
function getTouchPos(e) 
{
    var e = e || event;
    if (e.touches && e.touches.length == 1) {
        var touch = e.touches[0];
        touchX = touch.pageX - touch.target.offsetLeft;
        touchY = touch.pageY - touch.target.offsetTop;
    }
}
// getTouchPos(e): Helper function to get touch position relative to the canvas.
// Handles touch events, ensuring only one touch is considered (e.touches.length == 1).
// Calculates touch coordinates relative to the canvas using touch.pageX, touch.pageY, and subtracting offsets.





//-----------------------------------------------------------------------------------------------------------------//






// Function to get touch position relative to the canvas
function predict() 
{
    var modelEndpoint;
    var selectedModel = document.getElementById('model-select').value; // Get selected model from dropdown
    switch (selectedModel) 
    {
        case 'lenet':
            modelEndpoint = '/predict_lenet'; // Endpoint for LeNet model prediction
            break;
        case 'mlp':
            modelEndpoint = '/predict_mlp'; // Endpoint for mlp model prediction
            break;
        case 'logistic':
            modelEndpoint = '/predict_logistic'; // Endpoint for logistic model prediction
            break;
        default:
            return;
    }

    var imageData = canvas.toDataURL(); // Get image data from canvas as base64 URL

    fetch(modelEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }, // Send image data to server for prediction
        body: JSON.stringify({ image_data: imageData })
    })
    .then(response => response.json()) // Parse response as JSON
    .then(data => {
        const predictedDigit = data.results.indexOf(Math.max(...data.results)); // Determine predicted digit
        document.getElementById('prediction-result').innerText = `Predicted Digit: ${predictedDigit}`; // Display predicted digit

        updateChart(data.results); // Update chart with confidence percentages
    })
    .catch(error => console.error('Error:', error)); // Handle errors

}
// predict() Function: Initiates prediction based on the selected model.
// Determines the API endpoint (modelEndpoint) based on the selected model from a dropdown.
// Converts canvas image data to a base64 URL (imageData).
// Performs a POST request using fetch() to send imageData to the server for prediction.
// Handles the response by parsing it as JSON, determines the predicted digit, updates the prediction result element (prediction-result), and updates the chart using updateChart() with confidence percentages.




//-----------------------------------------------------------------------------------------------------------------//





// Event listener for predict button click
document.getElementById('predict_button').addEventListener('click', function() {
    predict(); // Call predict function
});

// Event listener for clear button click
document.getElementById('clear_button').addEventListener("click", function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
    ctx.fillStyle = "black"; // Set background color to black
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill canvas with black color
    document.getElementById('prediction-result').innerText = ''; // Clear prediction result
});
// Event Listeners: These listeners are set up for two buttons:
// Predict Button: Calls predict() when clicked, initiating the prediction process.
// Clear Button: Clears the canvas (ctx.clearRect()) and resets it to black (ctx.fillRect()), also clears the prediction result element (prediction-result).





//-----------------------------------------------------------------------------------------------------------------//






// Function to update the bar chart with confidence percentages
function updateChart(confidenceArray) {
    if (!barChart) 
    {
         // Initialize chart data and options
        var chartData = {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            datasets: [{
                label: 'Confidence Percentage',
                backgroundColor: confidenceArray.map((confidence, index) => generateColor(index)),
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                data: confidenceArray.map(confidence => confidence * 100)
            }]
        };

        var chartOptions = {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true,
                        min: 0,
                        max: 100,
                        callback: function(value) { return value + '%'; }
                    }
                }]
            }
        };
         
        // Create a new bar chart instance
        barChart = new Chart(document.getElementById('barChart'), {
            type: 'bar',
            data: chartData,
            options: chartOptions
        });
    } 
    // Update existing chart with new data
    else 
    {
        barChart.data.datasets[0].data = confidenceArray.map(confidence => confidence * 100);
        barChart.data.datasets[0].backgroundColor = confidenceArray.map((confidence, index) => generateColor(index));
        barChart.update(); // Update chart
    }
}
// updateChart(confidenceArray): Updates or initializes the bar chart with confidence percentages.
// If barChart does not exist, initializes it with chart data (chartData) and options (chartOptions).
// If barChart already exists, updates its datasets with new confidence percentages (confidenceArray), recalculates background colors using generateColor(index), and calls barChart.update() to redraw the chart.





//-----------------------------------------------------------------------------------------------------------------//






// Function to generate color for chart bars
function generateColor(number) 
{
    var colors = [
        'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 'rgba(255, 206, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)', 'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)',
        'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 'rgba(255, 206, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)'
    ];
    return colors[number % colors.length]; // Return color based on index
}
// generateColor(number): Helper function to generate colors for chart bars.
// Uses a predefined array (colors) of RGBA values with transparency (0.5).
// Returns a color from the array based on the index (number % colors.length), ensuring the color repeats if the index exceeds the array length.




//-----------------------------------------------------------------------------------------------------------------//






// Initialize the application when DOM content is loaded
document.addEventListener("DOMContentLoaded", function() {
    init(); // Initialize canvas and drawing context
    updateChart(confidencePercentages); // Update chart with initial confidence percentages
});
// DOMContentLoaded Event: Ensures that the initialization (init()) and initial chart update (updateChart(confidencePercentages)) occur after the DOM content is fully loaded.
// Calls init() to set up canvas and event listeners.
// Calls updateChart() to initialize the chart with initial confidence percentages (all zeros).