# 🏪 Retail Sales Forecasting & Inventory Optimization System

A Data Analytics & Time-Series Forecasting project focused on analyzing multi-store retail sales data and predicting future demand using Holt-Winters Exponential Smoothing.

---

## 🚀 Project Overview

This project aims to build a retail demand forecasting system capable of analyzing historical weekly sales data and predicting future sales trends.

The project follows a structured data analytics pipeline including:

- Data Exploration  
- Data Preprocessing  
- Seasonal Trend Analysis  
- Economic Impact Analysis  
- Store Performance Evaluation  
- Time-Series Forecasting  
- Model Evaluation  

---

## 🛠 Tech Stack & Libraries

* **Language:** Python  
* **Data Analysis:** Pandas, NumPy  
* **Visualization:** Seaborn, Matplotlib  
* **Time-Series Modeling:** Holt-Winters (Exponential Smoothing)  
* **Model Evaluation:** MAE, RMSE  
* **Statistical Analysis:** Correlation Analysis  

---

## 📊 Key Features & Methodology

To ensure reliable sales forecasting performance, the following techniques were implemented:

### 📌 Data Exploration & Analysis

- Checked dataset structure, duplicates, and summary statistics  
- Converted Date column into datetime format  
- Extracted Month and Year features  
- Analyzed correlation between Weekly Sales and economic indicators  
- Visualized relationships using heatmaps  

### 📈 Seasonal Trend Identification

- Aggregated monthly average sales  
- Identified peak sales during November & December  
- Observed holiday-driven revenue spikes  
- Detected seasonal demand patterns  

### 💼 Store Performance Evaluation

- Identified **Top Performing Store** based on total historical sales  
- Identified **Worst Performing Store**  
- Calculated revenue gap between highest and lowest performing stores  
- Measured store-wise sensitivity to unemployment, temperature, and CPI  

### 📉 Economic Impact Analysis

Performed store-wise correlation analysis to evaluate impact of:

- **Unemployment Rate**
- **Temperature**
- **Consumer Price Index (CPI)**

Identified stores most negatively affected by unemployment.

### 🔮 Time-Series Forecasting

Applied **Holt-Winters Exponential Smoothing**:

- Additive Trend  
- Additive Seasonality  
- 52-week seasonal cycle  

Forecasted weekly sales for the **next 12 weeks for all stores**.

---

## 📈 Model Performance

The forecasting model was evaluated using:

- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Square Error)**  

The model successfully captures:

- Sales trend  
- Seasonal fluctuations  
- Holiday demand spikes  

---

## 📁 Dataset

Multi-store Weekly Retail Sales Dataset  

**Target Variable:**  
`Weekly_Sales`

**Key Features:**

- Store  
- Date  
- Temperature  
- CPI  
- Unemployment  
- Fuel Price  
- Holiday Flag  

---

## ▶ How to Run the Project

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
python WalmartSalesProject.py
