from flask import Flask, render_template, request, url_for
import os
import pickle
import dill
import numpy as np
from skimage import io
from werkzeug.utils import secure_filename

from models.ExtractFeature import FeatureExtractor

app = Flask(__name__, template_folder="template", static_folder="static")

with open(os.path.join("models", "model_xgb.pkl"), "rb") as f:
    model_xgb = pickle.load(f)
with open(os.path.join("models", "model_isol.pkl"), "rb") as f:
    model_isol = pickle.load(f)
with open(os.path.join("models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
    
with open(os.path.join("models", "lime_explainer.pkl"), "rb") as f:
    lime_explainer = dill.load(f)

class CustomUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FeatureExtractor":
            return FeatureExtractor
        return super().find_class(module, name)

def custom_load(file_obj):
    return CustomUnpickler(file_obj).load()

with open(os.path.join("models", "feature_extractor.pkl"), "rb") as f:
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

    if pred_encoded == 0:
        classification = "BASOPHIL"
    elif pred_encoded == 1:
        classification = "EOSINOPHIL"
    elif pred_encoded == 2:
        classification = "ERYTHROBLAST"
    elif pred_encoded == 3:
        classification = "IG"
    elif pred_encoded == 4:
        classification = "LYMPHOCYTE"
    elif pred_encoded == 5:
        classification = "MONOCYTE"
    elif pred_encoded == 6:
        classification = "NEUTROPHIL"
    elif pred_encoded == 7:
        classification = "PLATELET"
    else:
        classification = "UNKNOWN... Please try again."

    anomaly_score = model_isol.decision_function(features_scaled)[0]
    anomaly_explanation = "Normal" if anomaly_score >= 0 else "Potentially Anomalous"

    exp = lime_explainer.explain_instance(features_scaled[0], model_xgb.predict_proba, num_features=15935)
    lime_explanation = exp.as_list()

    group_sums = {}
    group_counts = {}

    def get_feature_group(feature_name):
        fname = feature_name.lower()
        if "color_hist" in fname:
            return "Color Histogram"
        elif "hog_" in fname:
            return "HOG"
        elif "lbp" in fname:
            return "LBP"
        elif "gabor" in fname:
            return "Gabor"
        elif "gist" in fname:
            return "GIST"
        elif "hu_" in fname:
            return "Hu Moments"
        elif "zernike" in fname:
            return "Zernike Moments"
        elif "wavelet" in fname:
            return "Wavelet"
        elif "haralick" in fname:
            return "Haralick"
        else:
            return "Other"

    for feat_desc, score in lime_explanation:
        group = get_feature_group(feat_desc)
        group_sums[group] = group_sums.get(group, 0) + abs(score)
        group_counts[group] = group_counts.get(group, 0) + 1

    all_groups = ["Color Histogram", "HOG", "LBP", "Gabor", "GIST", 
                "Hu Moments", "Zernike Moments", "Wavelet", "Haralick"]

    results_list = []
    for group in all_groups:
        total = group_sums.get(group, 0)
        count = group_counts.get(group, 0)
        avg = total / count if count > 0 else 0
        results_list.append((group, avg))

    results_list = sorted(results_list, key=lambda x: x[1], reverse=True)

    lime_text = "\n".join([f"{i+1}- {group} ({avg:.4f})" for i, (group, avg) in enumerate(results_list)])

    return render_template("result.html",
                           image_file=filename,
                           classification=classification,
                           anomaly_score=anomaly_score,
                           anomaly_explanation=anomaly_explanation,
                           lime_explanation=lime_text)

if __name__ == "__main__":
    app.run(debug=True)
