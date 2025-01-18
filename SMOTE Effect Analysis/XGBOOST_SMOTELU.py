# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score 
import matplotlib.pyplot as plt
import warnings
import numpy as np

# Suppress warnings
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

# Initialize XGBoost model
xgb_clf = XGBClassifier(
    random_state=42,
    n_estimators=800,
    max_depth=6,
    min_child_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,  
    use_label_encoder=False, 
    eval_metric='logloss'  
)

# Number of folds
n_splits = 5
fold = 1

# Store AUC scores
all_train_roc_aucs = []
all_test_roc_aucs = []

# Prepare for plotting ROC curve for both train and test
plt.figure(figsize=(10, 8))

# Total number of samples
n_samples = len(X_train_full)

# Manually create train-validation splits
indices = np.arange(n_samples)
np.random.shuffle(indices)
split_size = n_samples // n_splits

# Manually create train-validation splits
for fold in range(1, n_splits + 1):
    print(f"\n========== Fold {fold} Başlıyor ==========\n")
    
    # Define the training and validation sets for the current fold
    val_start_idx = (fold - 1) * split_size
    val_end_idx = val_start_idx + split_size if fold != n_splits else n_samples
    val_indices = indices[val_start_idx:val_end_idx]
    train_indices = np.setdiff1d(indices, val_indices)

    X_train, X_val = X_train_full[train_indices], X_train_full[val_indices]
    y_train, y_val = y_train_full.iloc[train_indices], y_train_full.iloc[val_indices]

    # Train the model
    xgb_clf.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred_probs = xgb_clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred_probs)
    all_train_roc_aucs.append(train_auc)

    # Evaluate on validation set
    y_val_pred_probs = xgb_clf.predict_proba(X_val)[:, 1]
    y_val_pred = xgb_clf.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_pred_probs)
    all_test_roc_aucs.append(val_auc)

    # Print results
    print(f"Fold {fold} - Training ROC AUC Score: {train_auc:.2f}")
    print(f"Fold {fold} - Validation ROC AUC Score: {val_auc:.2f}")
    print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Plot ROC curve for the current fold
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_probs)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_probs)

    plt.plot(fpr_train, tpr_train, color='b', lw=2, label=f'Train Fold {fold} (AUC = {train_auc:.2f})')
    plt.plot(fpr_val, tpr_val, color='r', lw=2, label=f'Validation Fold {fold} (AUC = {val_auc:.2f})')

# Final Results
print("\n================== Çapraz Doğrulama Sonuçları ==================")
print(f"Ortalama Eğitim ROC AUC Skoru: {sum(all_train_roc_aucs) / len(all_train_roc_aucs):.2f}")
print(f"Ortalama Doğrulama ROC AUC Skoru: {sum(all_test_roc_aucs) / len(all_test_roc_aucs):.2f}")
print("===============================================================\n")

# Now evaluate on the test set

# Standardize the test data (same scaler used for training data)
X_test = test_data.drop(columns=['Attrition'])
y_test = test_data['Attrition']
X_test = scaler.transform(X_test)

# Make predictions on the test set
y_test_pred_probs = xgb_clf.predict_proba(X_test)[:, 1]
y_test_pred = xgb_clf.predict(X_test)

# Compute ROC AUC for the test set
test_auc = roc_auc_score(y_test, y_test_pred_probs)

# Print classification report and AUC score for test set
print("\nTest Set ROC AUC Score:", test_auc)
print("\nTest Set Classification Report:\n", classification_report(y_test, y_test_pred))

# Plot the ROC curves of training, validation, and test datasets
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(fpr_test, tpr_test, color='g', lw=2, label=f'Test Set (AUC = {test_auc:.2f})')

# Show the plot
plt.title("ROC Curves for Training, Validation, and Test Sets (All Folds)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.show()
