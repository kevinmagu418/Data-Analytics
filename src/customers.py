# churn_prediction_full.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\data\Q2 Dataset.csv"
OUTPUT_DIR = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Raw dataset shape:", df.shape)

# Strip whitespace from string columns
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Map Churn to binary target
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# Feature Engineering
# -----------------------------
# Tenure group
def tenure_group(t):
    if t <= 12: return "0-1 Year"
    elif t <= 24: return "1-2 Years"
    elif t <= 48: return "2-4 Years"
    else: return "4+ Years"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

# Create service flags
service_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]

def service_flag(x):
    if pd.isna(x): return 0
    x = str(x).lower()
    if x in ["no","no internet service","no phone service","none"]: return 0
    return 1

for s in service_cols:
    if s in df.columns:
        df[s + "_flag"] = df[s].apply(service_flag)

df["NumActiveServices"] = df[[c + "_flag" for c in service_cols if c + "_flag" in df.columns]].sum(axis=1)

# Contract mapping
if "Contract" in df.columns:
    df["Contract_ord"] = df["Contract"].map({"Month-to-month":0, "One year":1, "Two year":2})

# Payment simplification
if "PaymentMethod" in df.columns:
    df["PaymentMethod_simp"] = df["PaymentMethod"].str.replace("Electronic Check","E-Check", case=False)

# -----------------------------
# Quick EDA plots
# -----------------------------
sns.set(style="whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="ChurnFlag", data=df)
plt.title("Churn Distribution")
plt.xlabel("ChurnFlag (0=Stayed, 1=Churned)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "churn_distribution.png"))
plt.show()

# 2. MonthlyCharges by Churn
if "MonthlyCharges" in df.columns:
    plt.figure(figsize=(7,4))
    sns.boxplot(x="ChurnFlag", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges by Churn")
    plt.xlabel("ChurnFlag (0=Stayed, 1=Churned)")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_charges_by_churn.png"))
    plt.show()

# 3. Tenure distribution by Churn
if "tenure" in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x="tenure", hue="ChurnFlag", kde=True, element="step")
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tenure_by_churn.png"))
    plt.show()

# -----------------------------
# Prepare features & target
# -----------------------------
drop_cols = ["customerID"] if "customerID" in df.columns else []
X = df.drop(columns=drop_cols + ["Churn","ChurnFlag"], errors="ignore")
y = df["ChurnFlag"]

# One-hot encode categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Ensure all numeric
X = X.select_dtypes(include=np.number)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SMOTE on training set only
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE -> X_train_res:", X_train_res.shape, "y_train_res:", np.bincount(y_train_res))

# -----------------------------
# Train XGBoost
# -----------------------------
model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X_train_res.columns).sort_values(ascending=False)
top_features = importances.head(15)
plt.figure(figsize=(8,6))
top_features.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.show()

print("\nTop features by importance:\n", top_features)
