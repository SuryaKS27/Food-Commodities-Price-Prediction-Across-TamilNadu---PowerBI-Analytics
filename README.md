# 🌾 Food-Commodities-Price-Prediction-Across-TamilNadu---PowerBI-Analytics

## 📌 Abstract

This project focuses on the **real-time prediction and analysis of food commodity prices** in Tamil Nadu using **historical agricultural data** and **advanced machine learning models**. Leveraging government-regulated market data, the system uses **Prophet** and **Random Forest Regressor (RFR)** for accurate forecasting. 

Two interactive platforms are developed:
- 📊 **Power BI Dashboard** for exploring historical trends and comparative insights.
- 💻 **Dash Web Application** for real-time price prediction and visualization.

This dual approach empowers **farmers, traders, and policymakers** with transparent, timely, and actionable insights.

---

## 🎯 Aim

To develop a robust and user-friendly system for:
- Accurate real-time **forecasting of food commodity prices**.
- Interactive visualization of trends.
- Enabling **informed decision-making** to promote **market stability and efficiency**.

### ✅ Key Objectives

1. **Accurate Price Forecasting** using Prophet and Random Forest Regressor.
2. **Real-Time Insights** using Dash to dynamically visualize predictions.
3. **Interactive Dashboards** using Power BI for intuitive data exploration.
4. **Support Market Efficiency** by helping stakeholders manage price volatility.

---

## 📂 Dataset

- Source: [CEDA Agrimarket](https://ceda.ashoka.edu.in/)
- Period: 2020 to 2023
- Size: ~165,000 rows, 19 columns
- Features: `date`, `district`, `market`, `commodity`, `modal_price`, etc.

---

## 🧹 Data Preprocessing

Key preprocessing steps:
- ❌ Removed irrelevant columns (`id`, `state_id`, `market_id`)
- 🔍 Handled missing values and outliers
- 📆 Split datetime into components
- 🏷️ Engineered `price-range` feature
- 📁 Stored cleaned yearly data separately and merged into one master `.csv`

---

## 🛠️ Database Integration

- **MySQL (XAMPP)** used to store prediction results
- Three tables created:
  - `forecast_results_prophet`
  - `forecast_results_rf`
  - `market_prices`

---

## 🔮 Prediction Workflow

1. Group data by `district`, `market`, and `commodity`
2. **Prophet Model**:
   - Trained on `date`, predicted `modal_price`
   - Updated results in `forecast_results_prophet`
3. **Random Forest Regressor**:
   - Used datetime features like year/month
   - Predicted `modal_price`, updated `forecast_results_rf`
   - Evaluated with **RMSE** and **R² Score**

---

## 📈 Real-Time Visualization

### Power BI

Steps:
1. Connect XAMPP MySQL to Power BI via ODBC
2. Configure DSN in `ODBC Data Source Administrator`
3. Load and visualize commodity trends, comparisons, and forecasts

### Dash Web App

- Integrated ML models for **real-time prediction**
- Displays dynamic graphs for selected commodities and markets

---

## 🚀 Features

- ✅ Accurate Forecasts (Prophet + RFR)
- 📊 Real-Time Visualizations (Dash)
- 📉 Historical Trend Analysis (Power BI)
- 🌐 Localhost DB via MySQL for dynamic updates
- 🧠 Data-driven Decision Support

---

## 📌 Future Work

- 🔗 Incorporate external factors like **weather data**, **global trends**
- 🌍 Expand to other **states and commodities**
- 📱 Deploy Dash app on cloud (e.g., Heroku or AWS)
- 🔒 Add user authentication and admin panel for dashboard access

---

## 🤝 Stakeholder Benefits

- 👨‍🌾 **Farmers**: Better crop planning and selling strategies
- 🛒 **Traders**: Anticipate price trends for procurement decisions
- 🏛️ **Policymakers**: Monitor market conditions and intervene timely

---

## 📎 Tech Stack

- **Language**: Python
- **ML Models**: Prophet, Random Forest Regressor
- **Visualization**: Dash, Power BI
- **Database**: MySQL via XAMPP
- **Data Source**: CEDA Agrimarket (2020–2023)

---
