import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

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

indices = np.arange(len(feature_vector))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=rs, stratify=y_encoded)

X_train_idx = X[train_idx]
X_test_idx = X[test_idx]
y_train_idx = y_encoded[train_idx]
y_test_idx = y_encoded[test_idx]

test_df = feature_vector.iloc[test_idx].copy()

X_train_idx_clipped = np.clip(X_train_idx, -1e10, 1e10)
X_test_idx_clipped = np.clip(X_test_idx, -1e10, 1e10)

iso_forest = IsolationForest(contamination=0.05, random_state=rs)
iso_forest.fit(X_train_idx_clipped)
anomaly_scores = iso_forest.decision_function(X_test_idx_clipped)
anomaly_preds = iso_forest.predict(X_test_idx_clipped)

test_df['anomaly_score'] = anomaly_scores

print(f"Anomaly prediction distribution (value, count): {np.unique(anomaly_preds, return_counts=True)}")