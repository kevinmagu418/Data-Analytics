# movie_tv_exploration.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\data\Moviedbnw.csv"
OUTPUT_DIR = r"C:\Users\LIVEWAVE\Downloads\DataAnalytics\src\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.replace(" ", "").str.lower()
print("Raw dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -----------------------------
# 2. Text cleaning for titles/descriptions
# -----------------------------
for col in ["moviename", "scrapedname"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# -----------------------------
# 3. Numeric and date processing
# -----------------------------
numeric_cols = ["budget", "revenue", "rating", "directorsrating", "writersrating", "totalfollowers"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Average critic rating
df["averagecriticrating"] = df[["directorsrating", "writersrating"]].mean(axis=1)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if "year" not in df.columns:
    df["year"] = df["date"].dt.year

# -----------------------------
# 4. Genre extraction from OtherInfo
# -----------------------------
def extract_genre(info):
    if pd.isna(info):
        return np.nan
    genres_list = ["Action", "Adventure", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi",
                   "Thriller", "Animation", "Fantasy", "Family", "Documentary"]
    info = str(info)
    found = [g for g in genres_list if re.search(r'\b{}\b'.format(g), info, re.IGNORECASE)]
    return found[0] if found else np.nan

df["genre"] = df["otherinfo"].apply(extract_genre)

# -----------------------------
# 5. Exploratory Data Analysis (EDA)
# -----------------------------
sns.set(style="whitegrid")

# 5.1 Critics vs Audience rating correlation
plt.figure(figsize=(7,5))
sns.scatterplot(x="averagecriticrating", y="rating", data=df)
plt.title("Audience Rating vs Average Critic Rating")
plt.xlabel("Average Critic Rating")
plt.ylabel("Audience Rating")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "critics_vs_audience.png"))
plt.show()

corr = df[["averagecriticrating","rating"]].corr().iloc[0,1]
print(f"Correlation between critic and audience rating: {corr:.2f}")

# 5.2 Top genres by average audience rating
if "genre" in df.columns:
    genre_avg = df.groupby("genre")["rating"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(x=genre_avg.values, y=genre_avg.index, palette="pastel")
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.title("Top 10 Genres by Average Audience Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_genres.png"))
    plt.show()

# 5.3 Ratings trend over years (gradient line with colorbar)
if "year" in df.columns and "rating" in df.columns:
    df_year = df.dropna(subset=["year", "rating"])
    year_avg = df_year.groupby("year")["rating"].mean()
    if not year_avg.empty:
        plt.figure(figsize=(10,5))
        norm = plt.Normalize(year_avg.min(), year_avg.max())
        cmap = plt.cm.viridis

        for i in range(len(year_avg)-1):
            plt.plot(year_avg.index[i:i+2], year_avg.values[i:i+2],
                     color=cmap(norm(year_avg.values[i])), marker="o", markersize=6)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="Average Rating")

        plt.title("Average Audience Rating by Release Year")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ratings_by_year_colormap.png"))
        plt.show()

# 5.4 Budget vs Revenue vs Rating (bubble plot)
if {"budget", "revenue", "rating"}.issubset(df.columns):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x="budget", y="revenue", size="rating", hue="rating",
        data=df, sizes=(20,300), palette="viridis", alpha=0.7
    )
    plt.title("Budget vs Revenue (bubble size = Rating)")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "budget_revenue_rating.png"))
    plt.show()

# -----------------------------
# 6. Top directors by average rating
# -----------------------------
if "director" in df.columns:
    top_directors = df.groupby("director")["rating"].mean().sort_values(ascending=False).head(10)
    print("Top 10 directors by average rating:\n", top_directors)

# -----------------------------
# 7. Save cleaned dataset
# -----------------------------
df.to_csv(os.path.join(OUTPUT_DIR, "mobiedb_cleaned.csv"), index=False)
print("Cleaned dataset saved to outputs/mobiedb_cleaned.csv")
print("\nEDA complete. Plots saved to:", OUTPUT_DIR)
