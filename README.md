# Insurance Prediction Analysis and Dashboard

This repository contains a comprehensive analysis and predictive modeling project aimed at forecasting insurance expenses based on personal attributes such as age, BMI, smoking status, and region. The project uses multiple machine learning techniques, exploratory data analysis (EDA), and a user-friendly interactive dashboard built with Streamlit.

---

## Project Overview

### Objective
- Analyze factors affecting medical expenses.
- Build predictive models to forecast yearly insurance costs.
- Compare the performance of multiple machine learning models.
- Create an interactive dashboard for predictions and visual insights.

### Dataset
-**Source**:ExcelR
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
- **`insurance.csv`**: The original dataset.
- **`notebook.ipynb`**: The Jupyter Notebook containing the entire analysis and modeling workflow.
- **`app.py`**: The Streamlit application code for the interactive dashboard.
- **`presentation.pptx`**: PowerPoint slides summarizing the project.
- **`README.md`**: This file detailing the project.

---

## How to Run the Project

### Prerequisites
- Python 3.7+
- Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Run the Streamlit App
1. Navigate to the project directory.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. Open the displayed URL in your web browser.

---

## Results and Insights
- **Key Findings**:
  - Smoking is the most significant factor influencing medical expenses.
  - Random Forest achieved the best prediction performance with an R² of 0.928.
- **Use Case**:
  - Predict medical expenses for insurance underwriting or personal expense forecasting.

---

## Authors
- Harsh Patil


---

