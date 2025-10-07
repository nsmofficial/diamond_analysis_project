# 💎 Advanced Diamond Analysis & Forecasting System

A sophisticated web application built with Streamlit for analyzing diamond price data. This system leverages machine learning (XGBoost) to provide accurate forecasts for both individual diamonds and entire market categories, addressing specific business needs for price forecasting in the diamond industry.

The system moves beyond simple historical analysis to provide a true forecasting engine, predicting the next potential discount for a diamond based on its most recent characteristics and market context.

## ✨ Key Features

### 🔮 **Individual Diamond Forecasting**
- Provides a "next price" forecast for each unique diamond
- Complete with detailed explanations of factors driving the prediction
- Uses advanced ML models for accurate predictions

### 📊 **Category-Level Forecasting**
- Generates market-wide discount forecasts for entire diamond categories
- Examples: "0.50-0.59 D VVS1 3EX" category analysis
- Offers valuable business intelligence for market trends

### 🧠 **Explainable AI (XAI)**
- Uses SHAP (SHapley Additive exPlanations) to break down predictions
- Builds trust and provides actionable insights
- Transparent model decision-making process

### 📈 **Historical Analysis Suite**
- Full set of tools for analyzing historical trends
- Category performance analysis
- Feature correlations and market insights

### 🖥️ **Interactive Dashboard**
- User-friendly web interface
- Easy data upload and exploration
- Real-time visualization of results

### 📋 **Exportable Reports**
- All forecasts can be downloaded as professionally formatted Excel files
- Customizable report generation

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Web Framework** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Explainability** | SHAP |
| **Data Visualization** | Plotly, Matplotlib |
| **Statistics** | SciPy |

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
Make sure you have **Python 3.11 or newer** installed on your system.

### 2. Clone the Repository
Clone this project repository to your local machine (or simply download the source code).

```bash
git clone <repository-url>
cd diamond_analysis_project
```

### 3. Create Virtual Environment

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

```bash
pip install -r requirements.txt
```

## ▶️ How to Run the Application

Once the setup is complete, you can run the Streamlit application with a single command:

```bash
streamlit run main.py
```

Your web browser should automatically open to the application's dashboard at `http://localhost:8501`.

## 📊 Model Performance

### ⚠️ Important Note on MAPE (Mean Absolute Percentage Error)

In an earlier version of this project, the Mean Absolute Percentage Error (MAPE) was displayed as a performance metric. This has been **intentionally removed**.

#### The Issue
The MAPE formula involves division by the actual value: `|(Actual - Predicted) / Actual|`. This calculation fails and produces an infinitely large, meaningless result if the Actual discount value is 0%. Since it is common for diamonds to have a 0% discount, MAPE is not a robust or reliable metric for this specific business problem.

#### The Solution
The model's performance is now measured using **Mean Absolute Error (MAE)**. MAE measures the average error in absolute terms (e.g., "the model is off by an average of 1.71%") and does not have the "division-by-zero" problem, making it a much more stable and interpretable metric for this dataset.

The model's excellent MAE score is a true indicator of its high accuracy.

## 📁 Project Structure

```
diamond_analysis_project/
├── main.py                 # Main Streamlit application
├── demo.py                 # Demo script
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── sample-data.xlsx      # Sample data files
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ for the diamond industry**