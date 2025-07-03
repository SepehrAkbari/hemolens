import warnings
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

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

results = {}

lr = LogisticRegression(random_state=rs, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Logistic Regression'] = y_pred_lr

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
results['KNN'] = y_pred_knn

mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)
recall_knn = recall_score(y_test, y_pred_knn, average='weighted', zero_division=0)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted', zero_division=0)

svm_lin = SVC(kernel='linear', random_state=rs)
svm_lin.fit(X_train, y_train)
y_pred_svm_lin = svm_lin.predict(X_test)
results['SVM Linear'] = y_pred_svm_lin

mse_svm_lin = mean_squared_error(y_test, y_pred_svm_lin)
r2_svm_lin = r2_score(y_test, y_pred_svm_lin)
accuracy_svm_lin = accuracy_score(y_test, y_pred_svm_lin)
precision_svm_lin = precision_score(y_test, y_pred_svm_lin, average='weighted', zero_division=0)
recall_svm_lin = recall_score(y_test, y_pred_svm_lin, average='weighted', zero_division=0)
f1_svm_lin = f1_score(y_test, y_pred_svm_lin, average='weighted', zero_division=0)

svm_rbf = SVC(kernel='rbf', random_state=rs)
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
results['SVM RBF'] = y_pred_svm_rbf

mse_svm_rbf = mean_squared_error(y_test, y_pred_svm_rbf)
r2_svm_rbf = r2_score(y_test, y_pred_svm_rbf)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
precision_svm_rbf = precision_score(y_test, y_pred_svm_rbf, average='weighted', zero_division=0)
recall_svm_rbf = recall_score(y_test, y_pred_svm_rbf, average='weighted', zero_division=0)
f1_svm_rbf = f1_score(y_test, y_pred_svm_rbf, average='weighted', zero_division=0)

rf = RandomForestClassifier(random_state=rs)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = y_pred_rf

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

adb = AdaBoostClassifier(random_state=rs)
adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)
results['AdaBoost'] = y_pred_adb

mse_adb = mean_squared_error(y_test, y_pred_adb)
r2_adb = r2_score(y_test, y_pred_adb)
accuracy_adb = accuracy_score(y_test, y_pred_adb)
precision_adb = precision_score(y_test, y_pred_adb, average='weighted', zero_division=0)
recall_adb = recall_score(y_test, y_pred_adb, average='weighted', zero_division=0)
f1_adb = f1_score(y_test, y_pred_adb, average='weighted', zero_division=0)

model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=rs)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
results['XGBoost'] = y_pred_xgb

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0)