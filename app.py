from flask import Flask, render_template, request
import pickle
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ===== LOAD MODEL SAFELY (TUPLE OR DICT) =====
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    model = data["model"]
    label_encoder = data["label_encoder"]
elif isinstance(data, (tuple, list)):
    model = data[0]
    label_encoder = data[1]
else:
    raise ValueError("Unsupported model.pkl format")
# ============================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        speed = float(request.form["speed"])
        infill = float(request.form["infill"])

        input_data = [[temperature, speed, infill]]

        pred_index = model.predict(input_data)[0]
        prediction = label_encoder.inverse_transform([pred_index])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

