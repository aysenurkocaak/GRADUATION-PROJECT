# Import necessary libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, RocCurveDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# File paths for train and test datasets
train_file_path = r'C:\\Users\\Monster\\Desktop\\TEZ\\IBM_TEZ_SON\\cleaned_Veriler_Smoted_Train.csv'
train_data = pd.read_csv(train_file_path)

test_file_path = r'C:\\Users\\Monster\\Desktop\\TEZ\\IBM_TEZ_SON\\cleaned_Veriler_Test.csv'
test_data = pd.read_csv(test_file_path)

# Feature and target separation
y_train_full = train_data['Attrition']  # Target for train
X_train_full = train_data.drop(columns=['Attrition'])  # Features for train

# Standardize the data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)

# Initialize LightGBM model
lgbm_clf = LGBMClassifier(
    random_state=42,
    n_estimators=500,
    max_depth=2,
    min_child_samples=30,
    feature_fraction=0.6,
    class_weight='balanced',
    lambda_l1=1.0,  # L1 düzenlileştirme
    lambda_l2=1.0   # L2 düzenlileştirme
)

# 5-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

all_train_roc_aucs = []
all_test_roc_aucs = []

for train_index, test_index in kf.split(X_train_full, y_train_full):
    print(f"\n========== Fold {fold} Başlıyor ==========\n")
    
    # Split data for the current fold
    X_train, X_val = X_train_full[train_index], X_train_full[test_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[test_index]
    
    # Train the model with early stopping
    lgbm_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
    )
    
    # Evaluate on training set
    y_train_pred_probs = lgbm_clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred_probs)
    all_train_roc_aucs.append(train_auc)
    
    # Evaluate on validation set
    y_val_pred_probs = lgbm_clf.predict_proba(X_val)[:, 1]
    y_val_pred = lgbm_clf.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_pred_probs)
    all_test_roc_aucs.append(val_auc)
    
    # Print results
    print(f"Fold {fold} - Training ROC AUC Score: {train_auc:.2f}")
    print(f"Fold {fold} - Validation ROC AUC Score: {val_auc:.2f}")
    print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_val, y_val_pred_probs)
    plt.title(f"ROC Curve - Fold {fold}")
    plt.show()
    
    fold += 1

# Final Results
print("\n================== Çapraz Doğrulama Sonuçları ==================")
print(f"Ortalama Eğitim ROC AUC Skoru: {sum(all_train_roc_aucs) / len(all_train_roc_aucs):.2f}")
print(f"Ortalama Doğrulama ROC AUC Skoru: {sum(all_test_roc_aucs) / len(all_test_roc_aucs):.2f}")
print("===============================================================\n")