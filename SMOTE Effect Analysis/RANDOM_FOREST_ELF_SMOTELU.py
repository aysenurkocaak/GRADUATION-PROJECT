# Import necessary libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import warnings

# f1 score : %88 çıkıyor
warnings.filterwarnings("ignore", category=UserWarning)

# File paths for train and test datasets
train_file_path = r'C:\\Users\\Monster\\Desktop\\TEZ\\IBM_TEZ_SON\\cleaned_Veriler_Smoted_Train.csv'
train_data = pd.read_csv(train_file_path)

test_file_path = r'C:\\Users\\Monster\\Desktop\\TEZ\\IBM_TEZ_SON\\cleaned_Veriler_Test.csv'
# Load the data
test_data = pd.read_csv(test_file_path)

# Feature and target separation
y_train_full = train_data['Attrition']  # Target for train
X_train_full = train_data.drop(columns=['Attrition'])  # Features for train
y_test = test_data['Attrition']  # Target for test
X_test = test_data.drop(columns=['Attrition'])  # Features for test

# Standardize the data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)  # Apply the same scaler to the test set

# Initialize Random Forest model
rf_clf = RandomForestClassifier(
    random_state=42,
    n_estimators=1000,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample'
)

# 5-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Store AUC scores
all_train_roc_aucs = []
all_val_roc_aucs = []
all_test_roc_aucs = []  # Store the AUC score for the test set

# Prepare for plotting ROC curve for both train, validation, and test
plt.figure(figsize=(10, 8))

for train_index, val_index in kf.split(X_train_full, y_train_full):
    print(f"\n========== Fold {fold} Başlıyor ==========\n")
    
    # Split data for the current fold
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    # Train the model
    rf_clf.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred_probs = rf_clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred_probs)
    all_train_roc_aucs.append(train_auc)
    
    # Evaluate on validation set
    y_val_pred_probs = rf_clf.predict_proba(X_val)[:, 1]
    y_val_pred = rf_clf.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_pred_probs)
    all_val_roc_aucs.append(val_auc)
    
    # Evaluate on the test set (after each fold)
    y_test_pred_probs = rf_clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_probs)
    all_test_roc_aucs.append(test_auc)
    
    # Print results
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
    
    # Plot ROC curve for the current fold on both train and validation data
    RocCurveDisplay.from_estimator(rf_clf, X_train, y_train, name=f'Train Fold {fold}', ax=plt.gca())
    RocCurveDisplay.from_predictions(y_val, y_val_pred_probs, name=f'Validation Fold {fold}', ax=plt.gca())
    
    fold += 1

# Final Results
print("\n================== Çapraz Doğrulama Sonuçları ==================")
print(f"Ortalama Eğitim ROC AUC Skoru: {sum(all_train_roc_aucs) / len(all_train_roc_aucs):.2f}")
print(f"Ortalama Doğrulama ROC AUC Skoru: {sum(all_val_roc_aucs) / len(all_val_roc_aucs):.2f}")
print("===============================================================\n")

# Plot the ROC curves of training, validation, and test datasets
plt.title("ROC Curves for Training, Validation, and Test Sets (All Folds)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.show()
