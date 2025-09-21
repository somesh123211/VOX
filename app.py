from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # allow frontend JS to call backend API

# -----------------------------
# Load model and dictionary
# -----------------------------
try:
    model = joblib.load("sign_language_model.pkl")
    word_dict = joblib.load("word_dictionary.pkl")
    inverted_word_dict = {v: k for k, v in word_dict.items()}
    print("Model and dictionary loaded successfully.")
except Exception as e:
    model = None
    inverted_word_dict = {}
    print("Error loading model or dictionary:", e)

# -----------------------------
# Mediapipe Hand detection
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/mute")
def mute():
    return render_template("mute.html")

@app.route('/both')
def both_page():
    return render_template('both.html')

@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/deaf")
def deaf_page():
    return render_template("deaf.html")

@app.route("/morefeatures")
def morefeatures_page():
    return render_template("morefeatures.html")



@app.route("/sign_to_text", methods=["POST"])
def sign_to_text():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    # Read image from request.files
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:  # single hand
                prediction = model.predict([landmarks])
                predicted_word = inverted_word_dict.get(prediction[0], "Unknown")
                return jsonify({"word": predicted_word})

    return jsonify({"word": "No sign detected"})

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
