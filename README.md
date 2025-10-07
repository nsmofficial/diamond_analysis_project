💎 Advanced Diamond Analysis & Forecasting System
This project is a sophisticated web application built with Streamlit for analyzing diamond price data. It leverages a machine learning model (XGBoost) to provide accurate forecasts for both individual diamonds and entire market categories, addressing specific business needs for price forecasting in the diamond industry.The system moves beyond simple historical analysis to provide a true forecasting engine, predicting the next potential discount for a diamond based on its most recent characteristics and market context.
✨ Key Features
Individual Diamond Forecasting: Provides a "next price" forecast for each unique diamond, complete with a detailed explanation of the factors driving the prediction. Category-Level Forecasting: Generates a market-wide discount forecast for entire categories of diamonds (e.g., "0.50-0.59 D VVS1 3EX"), offering valuable business intelligence. Explainable AI (XAI): Uses SHAP (SHapley Additive exPlanations) to break down each prediction, building trust and providing actionable insights. Historical Analysis Suite: Includes a full set of tools for analyzing historical trends, category performance, and feature correlations. Interactive Dashboard: A user-friendly web interface that allows for easy data upload and exploration of results.

Exportable Reports: All forecasts can be downloaded as professionally formatted Excel files.

🛠️ Technology Stack
Language: Python 3.11+
Web Framework: Streamlit
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-learn, XGBoost
Model Explainability: SHAP
Data Visualization: Plotly, Matplotlib
Statistics: SciPy


🚀 Setup and Installation
Follow these steps to get the application running on your local machine.

1. Prerequisites
Make sure you have Python 3.11 or newer installed on your system.

2. Clone the Repository
Clone this project repository to your local machine (or simply download the source code).


# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


4. Install Dependencies
Install all the required Python libraries using the requirements.txt file.
   pip install -r requirements.txt


▶️ How to Run the Application
Once the setup is complete, you can run the Streamlit application with a single command:
   streamlit run main.py
Your web browser should automatically open to the application's dashboard.



⚠️ An Important Note on MAPE (Mean Absolute Percentage Error)
In an earlier version of this project, the Mean Absolute Percentage Error (MAPE) was displayed as a performance metric. This has been intentionally removed.

The Issue: The MAPE formula involves division by the actual value: |(Actual - Predicted) / Actual|. This calculation fails and produces an infinitely large, meaningless result if the Actual discount value is 0%. Since it is common for diamonds to have a 0% discount, MAPE is not a robust or reliable metric for this specific business problem.

The Solution: The model's performance is now measured using Mean Absolute Error (MAE). MAE measures the average error in absolute terms (e.g., "the model is off by an average of 1.71%") and does not have the "division-by-zero" problem, making it a much more stable and interpretable metric for this dataset. The model's excellent MAE score is a true indicator of its high accuracy.