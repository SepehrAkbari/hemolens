import pandas as pd
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

rs = 450

original_data_path = '../Data/bloodcells_dataset'
cleaned_data_path = '../Data/bloodcells_dataset_cleaned'

cell_types = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

feature_vector = pd.read_csv('../Data/bloodcells_dataset_cleaned_features.csv')

feature_cols = [col for col in feature_vector.columns if col not in ['filename', 'label']]
X = feature_vector[feature_cols].values
y = feature_vector['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=rs, stratify=y_encoded)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names=feature_cols, 
    class_names=le.classes_, 
    discretize_continuous=True,
    random_state=rs
)

instance = X_test[rs]

exp = lime_explainer.explain_instance(instance, model_xgb.predict_proba, num_features=len(feature_cols))
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

print("Feature Contributions (in order):")
for i, (group, avg) in enumerate(results_list):
    print(f"{i+1}- {group} ({avg:.4f})")