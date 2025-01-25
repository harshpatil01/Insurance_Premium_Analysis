import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Insurance Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

# Original Cleaned Dataset
df_original = load_data()

# Encoding Categorical Variables for Modeling
def encode_categorical_data(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    df_encoded['sex'] = le.fit_transform(df_encoded['sex'])
    df_encoded['smoker'] = le.fit_transform(df_encoded['smoker'])
    df_encoded['region'] = le.fit_transform(df_encoded['region'])
    return df_encoded

# Cleaned Original Dataset (No Encoding)
def clean_original_data(df):
    df_cleaned = df.copy()
    return df_cleaned

df_encoded = encode_categorical_data(df_original)  # Encoded dataset for modeling
df_cleaned = clean_original_data(df_original)  # Cleaned dataset for display

# Scaling for Encoded Data
scaler = MinMaxScaler()
df_encoded[['age', 'bmi', 'children']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children']])

# Split Data for Training
X = df_encoded.drop('expenses', axis=1)
y = df_encoded['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_svm = SVR(kernel='linear')
model_svm.fit(X_train, y_train)

model_dt = DecisionTreeRegressor(max_depth=10)
model_dt.fit(X_train, y_train)

model_rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model_rf.fit(X_train, y_train)

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a Section", ("Data Overview", "Make Predictions", "Model Evaluation"))

if section == "Data Overview":
    st.title("📊 Insurance Dataset Overview")
    st.write("This section displays the original cleaned dataset, which contains details about individuals' medical expenses.")

    # Display Original Cleaned Dataset
    columns_to_show = st.multiselect("Select Columns to View", df_cleaned.columns, default=df_cleaned.columns.tolist())
    st.write(df_cleaned[columns_to_show])

    # Statistical Summary for Original Dataset
    if st.checkbox("Show Statistical Summary"):
        st.subheader("Statistical Summary (Original Data)")
        st.write(df_cleaned.describe(include='all'))

    # Missing Data
    if st.checkbox("Show Missing Data"):
        st.subheader("Missing Data")
        st.write(df_cleaned.isnull().sum())

    # Feature Distribution (User Can Choose Feature to Visualize)
    selected_feature = st.selectbox("Choose a feature to visualize distribution", df_cleaned.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_cleaned[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap for Encoded Dataset
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap (Encoded Data)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif section == "Make Predictions":
    st.title("💡 Make Predictions")
    st.write("Enter the details below to predict yearly medical expenses.")

    # Input Fields
    age = st.slider("Age", 18, 100, 30)
    sex = st.radio("Sex", ["Male", "Female"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.radio("Smoker Status", ["Yes", "No"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

    # Convert Inputs
    sex_encoded = 1 if sex == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0
    region_mapping = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
    region_encoded = region_mapping[region]

    # Prepare Input Data
    input_data_raw = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region': [region_encoded]
    })

    # Scale only the numerical features
    scaled_features = ['age', 'bmi', 'children']
    input_data_scaled = input_data_raw.copy()
    input_data_scaled[scaled_features] = scaler.transform(input_data_raw[scaled_features])

    # Model Selection
    model_choice = st.selectbox("Choose a Model", ("Linear Regression", "Support Vector Machine", "Decision Tree", "Random Forest"))

    if model_choice == "Linear Regression":
        if st.button("Predict Expenses with Linear Regression"):
            prediction = model_lr.predict(input_data_scaled)
            st.success(f"Predicted Yearly Expense: **${prediction[0]:,.2f}**")

    elif model_choice == "Support Vector Machine":
        if st.button("Predict Expenses with SVM"):
            prediction = model_svm.predict(input_data_scaled)
            st.success(f"Predicted Yearly Expense: **${prediction[0]:,.2f}**")

    elif model_choice == "Decision Tree":
        if st.button("Predict Expenses with Decision Tree"):
            prediction = model_dt.predict(input_data_scaled)
            st.success(f"Predicted Yearly Expense: **${prediction[0]:,.2f}**")

    elif model_choice == "Random Forest":
        if st.button("Predict Expenses with Random Forest"):
            prediction = model_rf.predict(input_data_scaled)
            st.success(f"Predicted Yearly Expense: **${prediction[0]:,.2f}**")

elif section == "Model Evaluation":
    st.title("📊 Model Evaluation")
    st.write("This section evaluates the performance of different models.")
    
    # User Input for Test Size
    test_size = st.slider("Select Test Size for Data Split", 0.1, 0.9, 0.2)

    # Re-split data with user-defined test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Evaluation Metrics
    lr_pred = model_lr.predict(X_test)
    svm_pred = model_svm.predict(X_test)
    dt_pred = model_dt.predict(X_test)
    rf_pred = model_rf.predict(X_test)

    evaluation_data = {
        "Model": ["Linear Regression", "Support Vector Machine", "Decision Tree", "Random Forest"],
        "Mean Squared Error (MSE)": [
            mean_squared_error(y_test, lr_pred),
            mean_squared_error(y_test, svm_pred),
            mean_squared_error(y_test, dt_pred),
            mean_squared_error(y_test, rf_pred)
        ],
        "R-squared (R²)": [
            r2_score(y_test, lr_pred),
            r2_score(y_test, svm_pred),
            r2_score(y_test, dt_pred),
            r2_score(y_test, rf_pred)
        ]
    }
    evaluation_df = pd.DataFrame(evaluation_data)
    st.write("### Model Comparison")
    st.write(evaluation_df)

    # Bar Plots for Metrics
    metric_choice = st.selectbox("Choose Metric to Visualize", ["Mean Squared Error (MSE)", "R-squared (R²)"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y=metric_choice, data=evaluation_df, ax=ax)
    plt.title(f"Model Comparison: {metric_choice}")
    st.pyplot(fig)

# Additional Styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .stButton button {
            background-color: #2d6a4f;
            color: white;
            font-weight: bold;
        }
        .stSlider div {
            background-color: #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)