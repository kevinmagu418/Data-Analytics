import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration

DATA_PATH = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\data\heart_disease.csv"
OUTPUT_DIR = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#  Load dataset

df = pd.read_csv(DATA_PATH)
print("Raw dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# 2. Clean string columns safely
for c in df.columns:
    if df[c].dtype == object:
        # Only apply .str.strip() if all non-NaN values are strings
        if df[c].dropna().map(lambda x: isinstance(x, str)).all():
            df[c] = df[c].str.strip()

# Convert boolean-like columns to actual bool
bool_cols = ['fbs', 'exang']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)

# Ensure numeric columns are numeric
numeric_cols = ['age','trestbps','chol','thalch','oldpeak','ca','num']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# 3Descriptive statistics

desc_stats = df.describe(include='all')
desc_stats.to_csv(os.path.join(OUTPUT_DIR, "heart_descriptive_stats.csv"))
print("Descriptive statistics saved.")


# Correlation analysis

numeric_corr = df[numeric_cols].corr()
numeric_corr.to_csv(os.path.join(OUTPUT_DIR, "heart_numeric_correlation.csv"))
print("Numeric correlations saved.")


# Visualizations

sns.set(style="whitegrid")

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', bins=20, kde=True, color='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"))
plt.close()

# Cholesterol distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='chol', bins=20, kde=True, color='black')
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chol_distribution.png"))
plt.close()

# Age vs Max Heart Rate scatterplot by outcome
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='age', y='thalch', hue='num', palette='viridis')
plt.title("Age vs Max Heart Rate by Outcome")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalch)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_vs_thalch.png"))
plt.close()

# Cholesterol by Sex boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='sex', y='chol', color='black')
plt.title("Cholesterol by Sex")
plt.xlabel("Sex")
plt.ylabel("Cholesterol")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chol_by_sex.png"))
plt.close()

print("All plots saved to:", OUTPUT_DIR)
print("Public Health & Demographics Analysis Complete.")
