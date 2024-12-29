from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Load the Keras model
MODEL_PATH = "model/fruits_and_vegetables_model.h5"  # Path to your .h5 model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load .h5 model: {str(e)}")

# Class labels
CLASS_NAMES = [
    "Apple", "Banana", "Cucumber", "Tomato", "Onion", "Broccoli", 
    "Corn", "Avocado", "Bell pepper", "Grapes", "Papaya", "Watermelon", 
    "Pineapple", "Pear", "Mango", "Lemon", "Zucchini", "Potato", "Carrot"
]

# Home route (optional: for a web UI)
@app.route("/")
def home():
    return render_template("index.html")

# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image file is provided
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Open and preprocess the image
        img = Image.open(file).convert("RGB")
        img = img.resize((150, 150))  # Match model input size
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run inference
        predictions = model.predict(img_array)
        
        # Find the predicted class
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        # Return the result as JSON
        return jsonify({
            "class": predicted_class,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8089)
