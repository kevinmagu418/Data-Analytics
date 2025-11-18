# crime_data_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
DATA_PATH = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\data\crimes.csv"
OUTPUT_DIR = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset

df = pd.read_csv(DATA_PATH)
print("Raw dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Data cleaning

# Standardize column names
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Convert date columns
df['date_rptd'] = pd.to_datetime(df['date_rptd'], errors='coerce')
df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')

# Convert time column to string for easier handling
df['time_occ'] = df['time_occ'].astype(str).str.zfill(4)

# Strip whitespace from string columns
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].str.strip()

# -----------------------------
# 3. Aggregated statistics
# -----------------------------
# Top areas by crime count
area_counts = df['area_name'].value_counts().head(10)
print("\nTop 10 areas by crime count:\n", area_counts)

# Top crime types
crime_type_counts = df['crm_cd_desc'].value_counts().head(10)
print("\nTop 10 crime types:\n", crime_type_counts)

# Monthly crime trend
df['month'] = df['date_occ'].dt.to_period('M')
monthly_counts = df.groupby('month').size()
print("\nMonthly crime counts (sample):\n", monthly_counts.head())

# Yearly trend
df['year'] = df['date_occ'].dt.year
yearly_counts = df.groupby('year').size()
print("\nYearly crime counts:\n", yearly_counts)


# 4. Plots

sns.set(style='whitegrid')

# 4.1 Top areas bar plot
plt.figure(figsize=(8,5))
sns.barplot(x=area_counts.values, y=area_counts.index, palette='viridis')
plt.xlabel("Number of Crimes")
plt.ylabel("Area")
plt.title("Top 10 Areas by Crime Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_areas.png"))
plt.show()

# 4.2 Top crime types
plt.figure(figsize=(8,5))
sns.barplot(x=crime_type_counts.values, y=crime_type_counts.index, palette='magma')
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.title("Top 10 Crime Types")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_crime_types.png"))
plt.show()

# 4.3 Monthly trend line plot
plt.figure(figsize=(10,5))
sns.lineplot(x=monthly_counts.index.astype(str), y=monthly_counts.values, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.title("Monthly Crime Counts")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_trend.png"))
plt.show()

# 4.4 Yearly trend
plt.figure(figsize=(8,5))
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='red')
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.title("Yearly Crime Counts")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yearly_trend.png"))
plt.show()


# 5. Save cleaned and aggregated data
df.to_csv(os.path.join(OUTPUT_DIR, "crime_data_cleaned.csv"), index=False)
area_counts.to_csv(os.path.join(OUTPUT_DIR, "top_areas.csv"))
crime_type_counts.to_csv(os.path.join(OUTPUT_DIR, "top_crime_types.csv"))
monthly_counts.to_csv(os.path.join(OUTPUT_DIR, "monthly_counts.csv"))
yearly_counts.to_csv(os.path.join(OUTPUT_DIR, "yearly_counts.csv"))

print("\nCleaned dataset and summary statistics saved to:", OUTPUT_DIR)
print("Analysis complete. Plots saved for report.")
