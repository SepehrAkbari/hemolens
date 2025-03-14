from flask import Flask, render_template, request, url_for
import os
import pickle
import dill
import numpy as np
from skimage import io
from werkzeug.utils import secure_filename

from Models.ExtractFeature import FeatureExtractor

app = Flask(__name__, template_folder="Template", static_folder="Static")

with open(os.path.join("Models", "model_xgb.pkl"), "rb") as f:
    model_xgb = pickle.load(f)
with open(os.path.join("Models", "model_isol.pkl"), "rb") as f:
    model_isol = pickle.load(f)
with open(os.path.join("Models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
    
with open(os.path.join("Models", "lime_explainer.pkl"), "rb") as f:
    lime_explainer = dill.load(f)

class CustomUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FeatureExtractor":
            return FeatureExtractor
        return super().find_class(module, name)

def custom_load(file_obj):
    return CustomUnpickler(file_obj).load()

with open(os.path.join("Models", "feature_extractor.pkl"), "rb") as f:
    feature_extractor = custom_load(f)

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    image = io.imread(file_path)

    features_dict = feature_extractor.extract_features_dict(image)
    feature_vector = np.array(list(features_dict.values())).reshape(1, -1)

    features_scaled = scaler.transform(feature_vector)

    pred_encoded = model_xgb.predict(features_scaled)[0]
    classification = str(pred_encoded)

    anomaly_score = model_isol.decision_function(features_scaled)[0]
    anomaly_explanation = "This cell appears normal." if anomaly_score >= 0 else "This cell appears potentially anomalous."

    exp = lime_explainer.explain_instance(features_scaled[0], model_xgb.predict_proba, num_features=10)
    lime_explanation = exp.as_list()
    lime_text = "\n".join([f"{desc}: {score:.4f}" for desc, score in lime_explanation])

    return render_template("result.html",
                           image_file=filename,
                           classification=classification,
                           anomaly_score=anomaly_score,
                           anomaly_explanation=anomaly_explanation,
                           lime_explanation=lime_text)

if __name__ == "__main__":
    app.run(debug=True)
