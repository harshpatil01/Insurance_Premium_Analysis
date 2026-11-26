# Insurance Prediction Analysis and Dashboard

> **Context:** Data science internship project at Ai Variant (2024–2025), built on a public dataset. Shared with no confidential company data.

This repository contains a comprehensive analysis and predictive modeling project aimed at forecasting insurance expenses based on personal attributes such as age, BMI, smoking status, and region. The project uses multiple machine learning techniques, exploratory data analysis (EDA), and a user-friendly interactive dashboard built with Streamlit.

---

## Project Overview

### Objective
- Analyze factors affecting medical expenses.
- Build predictive models to forecast yearly insurance costs.
- Compare the performance of multiple machine learning models.
- Create an interactive dashboard for predictions and visual insights.

### Dataset
- **Source**: Provided by ExcelR as part of the Data Science course. It is the same as the public [Medical Cost Personal Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).
- **Size**: 1338 rows and 7 columns
- **Features**:
  - `age`: Age of the individual.
  - `sex`: Gender of the individual.
  - `bmi`: Body Mass Index.
  - `children`: Number of children/dependents.
  - `smoker`: Smoking status (yes/no).
  - `region`: Residential region (northeast, northwest, southeast, southwest).
  - `expenses`: Yearly medical expenses (target variable).

---

## Steps in the Project

### 1. Data Preprocessing
- **Data Cleaning**:
  - Removed missing values (if any).
  - Verified the integrity of the dataset.
- **Encoding**:
  - Categorical features (`sex`, `smoker`, `region`) were label-encoded for machine learning.
- **Scaling**:
  - Numerical features (`age`, `bmi`, `children`) were normalized using MinMaxScaler.

### 2. Exploratory Data Analysis (EDA)
- **Feature Distribution**:
  - Visualized distributions of key features like `age`, `bmi`, and `expenses`.
- **Correlation Analysis**:
  - Created a heatmap to identify relationships between variables.
- **Insights**:
  - Smoking status has a significant impact on insurance expenses.
  - BMI shows a weak correlation with expenses.

### 3. Machine Learning Models
We implemented and evaluated the following models:

#### Linear Regression
- **Description**: A simple baseline regression model.
- **Performance**:
  - Mean Squared Error (MSE): 36,525,540
  - R-squared (R²): 0.750

#### Multiple Linear Regression
- **Description**: Extended regression model incorporating interaction terms.
- **Performance**:
  - MSE: 33,639,080
  - R²: 0.783

#### Decision Tree Regressor
- **Description**: A tree-based model capturing non-linear relationships.
- **Performance**:
  - MSE: 15,425,350
  - R²: 0.901

#### Random Forest Regressor
- **Description**: An ensemble model for robust predictions.
- **Best Parameters**: `{'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 5}`
- **Performance**:
  - Optimized MSE: 11,175,610
  - Optimized R²: 0.928

#### Support Vector Regression (SVR)
- **Description**: A kernel-based regression method.
- **Performance**:
  - MSE: 20,956,870
  - R²: 0.865

---

### 4. Model Evaluation
- Compared all models using:
  - Mean Squared Error (MSE)
  - R-squared (R²)
- Highlighted Random Forest as the best-performing model.

### 5. Streamlit Dashboard
- **Features**:
  - Data Overview: Displays original and cleaned datasets, summary statistics, and visualizations.
  - Prediction Tool: Allows users to input details (age, BMI, etc.) and predict expenses using a chosen model.
  - Model Evaluation: Compares performance metrics of implemented models.

---

## Repository Contents
All project files are in the `Insurance_Premium_Analysis/` folder:
- **`insurance.csv`**: The dataset.
- **`Insurance_Analysis.ipynb`**: The Jupyter Notebook with the full analysis and modelling workflow.
- **`app.py`**: The Streamlit dashboard.
- **`Insurance_Prediction_Model_Presentation.pdf`**: Slides summarising the project.
- **`requirements.txt`**: Python dependencies.

---

## How to Run the Project

### Prerequisites
- Python 3.9+

### Run the Streamlit App
```bash
git clone https://github.com/harshpatil01/Insurance_Premium_Analysis.git
cd Insurance_Premium_Analysis/Insurance_Premium_Analysis
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints in your browser.

---

## Results and Insights
- **Key Findings**:
  - Smoking is the most significant factor influencing medical expenses.
  - Random Forest achieved the best prediction performance with an R² of 0.928.
- **Caveats**:
  - R² values come from a single train/test split on a small dataset (1,338 rows), so they may vary with a different split.
  - This is a learning project on a public dataset, not a model used for real underwriting decisions.

---

## Authors
- Harsh Patil


---


## License

Code is released under the [MIT License](LICENSE). Datasets keep their original source licences.
