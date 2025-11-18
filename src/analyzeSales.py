import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATASET (Absolute Path)
df = pd.read_csv(r"C:/Users/LIVEWAVE/Downloads/DataAnalytics/src/data/Q1 Dataset.csv")


print("First 5 rows:")
print(df.head())

print("\nDataset Summary:")
print(df.info())


# DATA CLEANING

print("\nMissing values per column:")
print(df.isnull().sum())

df.fillna(0, inplace=True)      # Simple missing value fix
if "Category" in df.columns:
    df["Category"] = df["Category"].str.title().str.strip()


# FEATURE ENGINEERING

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()


# ANALYSIS

if "Product Name" in df.columns and "Sales" in df.columns:
    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop Products:")
    print(top_products)

if "Region" in df.columns and "Sales" in df.columns:
    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )
    print("\nSales by Region:")
    print(region_sales)

if "Month" in df.columns and "Sales" in df.columns:
    monthly_sales = df.groupby("Month")["Sales"].sum()


# VISUALS

sns.set(style="whitegrid")

# Sales by Region
if "region_sales" in locals():
    plt.figure(figsize=(8,5))
    sns.barplot(x=region_sales.values, y=region_sales.index)
    plt.title("Sales by Region")
    plt.show()

# Monthly Trend
if "monthly_sales" in locals():
    plt.figure(figsize=(8,5))
    monthly_sales.plot(kind="line", marker="o")
    plt.title("Monthly Sales Trend")
    plt.show()

# Top Products
if "top_products" in locals():
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title("Top 10 Best-Selling Products")
    plt.show()
