import pandas as pd
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP DAGSHUB & MLFLOW
# ==========================================
import dagshub
import os
import mlflow

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "karimaulya"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Ganti dengan URL DagsHub kamu
DAGSHUB_TRACKING_URI = "https://dagshub.com/karimaulya/Eksperimen_SML_Karima-Ulya-Hermawan.mlflow"
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
mlflow.set_experiment("Student_Depression_Prediction_Tuning")

# ==========================================
# 2. LOAD DATASET BERSIH
# ==========================================
data_path = "dataset_clean.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['Depression'])
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. MEMBANGUN MODEL & HYPERPARAMETER TUNING
# ==========================================
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="RandomForest_Tuning_Run"):
    print("Memulai proses Hyperparameter Tuning dengan GridSearchCV...")
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nParameter terbaik ditemukan: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # ==========================================
    # 4. MANUAL LOGGING KE MLFLOW
    # ==========================================
    mlflow.log_params(grid_search.best_params_)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # ==========================================
    # 5. MEMBUAT & MENYIMPAN ARTEFAK GAMBAR
    # ==========================================
    print("\nMembuat dan mengunggah artefak visualisasi ke DagsHub...")
    
    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_path = "screenshoot_artifak_1.jpg"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # Artefak 2: Feature Importance
    feature_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,6))
    sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    feat_path = "screenshoot_artifak_2.jpg"
    plt.tight_layout()
    plt.savefig(feat_path)
    mlflow.log_artifact(feat_path)
    plt.close()
    
    print("Selesai! Semua data, metrik, dan artefak berhasil dikirim ke DagsHub.")