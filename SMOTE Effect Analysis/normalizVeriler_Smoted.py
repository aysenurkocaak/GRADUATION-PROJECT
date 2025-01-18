from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. Dosyanın yüklenmesi
file_path = r'C:\Users\Monster\Desktop\TEZ\IBM_TEZ_SON\cleaned_veriler_yeni_normalized.csv'
data = pd.read_csv(file_path)

# 2. Özellikler (features) ve hedef değişkenin (target) ayrılması
X = data.drop(columns=['Attrition'])  # Özellik sütunları
y = data['Attrition']  # Hedef sütun

# 3. Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Orijinal sınıf dağılımını kontrol et
original_train_distribution = Counter(y_train)
original_test_distribution = Counter(y_test)
print("Orijinal eğitim seti sınıf dağılımı:", original_train_distribution)
print("Orijinal test seti sınıf dağılımı:", original_test_distribution)

# 5. SMOTE işlemi (sadece eğitim setine uygulanır)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 6. SMOTE sonrası sınıf dağılımını kontrol et
resampled_train_distribution = Counter(y_train_resampled)
print("SMOTE sonrası eğitim seti sınıf dağılımı:", resampled_train_distribution)

# 7. SMOTE uygulanmış eğitim setini kaydetme
X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
X_train_resampled_df['Attrition'] = y_train_resampled

output_train_path = r'C:\Users\Monster\Desktop\TEZ\IBM_TEZ_SON\cleaned_Veriler_Smoted_Train.csv'
X_train_resampled_df.to_csv(output_train_path, index=False)
print(f"SMOTE uygulanmış eğitim seti başarıyla kaydedildi: {output_train_path}")

# 8. Test setini kaydetme
X_test_df = X_test.copy()
X_test_df['Attrition'] = y_test

output_test_path = r'C:\Users\Monster\Desktop\TEZ\IBM_TEZ_SON\cleaned_Veriler_Test.csv'
X_test_df.to_csv(output_test_path, index=False)
print(f"Test seti başarıyla kaydedildi: {output_test_path}")