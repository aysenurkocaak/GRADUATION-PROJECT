import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np

# Veriyi yükle ve karıştır
data = pd.read_csv(r"C:\Users\Monster\Desktop\TEZ\IBM_TEZ_SON\cleaned_veriler_yeni_normalized.csv")
data = shuffle(data, random_state=42)

# Özellikleri ve hedef değişkeni ayır
X = data.drop("Attrition", axis=1)  # Hedef değişkenin adını buraya yaz
y = data["Attrition"]

# Veriyi stratify ile eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Sadece eğitim verisi ile scaler fit et, test verisine uygula
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli tanımla
clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# StratifiedKFold ile çapraz doğrulama
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    # Eğitim ve doğrulama verisini ayır
    X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Modeli eğit
    clf.fit(X_train_cv, y_train_cv)
    
    # Doğrulama setinde tahmin
    y_val_proba = clf.predict_proba(X_val_cv)[:, 1]
    auc = roc_auc_score(y_val_cv, y_val_proba)
    auc_scores.append(auc)

print(f"Cross-Validation AUC Scores: {auc_scores}")
print(f"Mean CV AUC: {np.mean(auc_scores):.2f}")

# Modeli tam eğitim verisinde eğit
clf.fit(X_train_scaled, y_train)

# Test seti tahminleri
y_test_pred = clf.predict(X_test_scaled)
y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Performans metrikleri
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)

print("\nTest Set Performance Metrics:")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")
print(f"Test AUC: {auc:.2f}")
