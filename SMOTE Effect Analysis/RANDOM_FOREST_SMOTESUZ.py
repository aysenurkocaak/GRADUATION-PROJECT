import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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
clf = RandomForestClassifier(random_state=42)

# Cross-validation ile değerlendirme
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.2f}")

# Modeli eğit
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

print("\nModel Performance on Test Set:")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")
print(f"Test AUC: {auc:.2f}")
