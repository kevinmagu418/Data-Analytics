# Data-Analytics

# 📊 Data Analytics Assignment

This repository contains solutions for **five data analytics exercises**, each focused on developing practical skills in data cleaning, analysis, visualization, geospatial insights, text analysis, and machine learning. The project is structured for clarity, reproducibility, and easy review.

---

## 🗂️ Repository Structure

```
DataAnalytics/
│
├── Exercise1_Sales_Analysis/
│   
├── Exercise2_Customer_Segmentation/
│   ├──
│
├── Exercise3_Text_Mining/
│   ├──
│
├── Exercise4_Geospatial_Insights/
│   ├── 
│
├── Exercise5_Predictive_Modeling/
│   ├──
│
├── requirements.txt
└── README.md
```

---

## 📘 Exercise Summaries

### **1. Sales Performance & Trend Analysis**

Focuses on analyzing sales data to identify trends, seasonal patterns, and top-performing products/regions.

* Tools: Pandas, NumPy, Matplotlib, Seaborn
* Tasks include data cleaning, summary statistics, and visual trend analysis.

---

### **2. Customer Segmentation**

Uses clustering techniques to group customers based on behavior and attributes.

* Tools: Scikit‑learn, Pandas, NumPy
* Includes preprocessing, scaling, clustering (K-Means), and cluster interpretation.

---

### **3. Text Mining & Sentiment Analysis**

Analyzes text data to extract insights and detect sentiment.

* Tools: NLTK or TextBlob, Pandas
* Covers tokenization, cleaning, word frequency, and sentiment scoring.

---

### **4. Geospatial Data Analysis**

Visualizes geospatial patterns such as regional trends or location‑based insights.

* Tools: Folium, Plotly (GeoPandas avoided for Python 3.14 compatibility)
* Includes map creation, choropleth visualization, and spatial interpretation.

---

### **5. Predictive Modeling**

Builds a machine learning model to predict future outcomes (e.g., sales, churn, classifications).

* Tools: Scikit‑learn
* Covers preprocessing, model training, evaluation, and performance metrics.

---

## 🧪 Running the Project

### **1. Create a Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Launch Jupyter Lab or Notebook**

```bash
jupyter lab
```

---

## 🔒 Avoiding Git Conflicts

To prevent merge conflicts:

* Always **pull** before you push:

  ```bash
  git pull origin main
  ```
* Commit only files inside exercise folders.
* Do **not** commit `venv/` (it should be in your `.gitignore`).
* Work on each exercise inside its respective subfolder.

---

## 📄 License

This project is for academic purposes. Feel free to fork or reference with credit.

---

## 🙌 Author

Developed by **Kevin Kiragu**.
