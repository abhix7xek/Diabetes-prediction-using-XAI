import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model_filename = "diabetes_model.pkl"  # Ensure the model is in the correct path
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Set up Streamlit configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Header
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This application uses a machine learning model to predict the likelihood of diabetes based on user-provided medical information.
Please fill in the values in the sidebar and click *Predict* to view the results.
""")

# Sidebar for user input with a better layout
st.sidebar.header("Enter Your Medical Information")

# User inputs
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
insulin = st.sidebar.slider("Insulin", 0, 846, 79)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 0.0, 67.1, 21.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
age = st.sidebar.slider("Age", 21, 100, 30)

# Display information about the user inputs
st.sidebar.markdown("""
Use the sliders to input your medical information, which will be used to predict the likelihood of diabetes.
Once done, click *Predict* to receive results.
""")

# Prediction
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display Results
    st.subheader("Prediction Result")
    
    # Pie chart for the prediction probability
    labels = ["No Diabetes", "Diabetes"]
    probabilities = prediction_proba[0]
    
    # Plotting the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#F44336"], explode=(0.1, 0), shadow=True)
    ax.set_title("Prediction Probability of Diabetes")
    st.pyplot(fig)

    # Display prediction message
    if prediction[0] == 1:
        st.error("The model predicts that the individual is *likely to have diabetes.*")
    else:
        st.success("The model predicts that the individual is *unlikely to have diabetes.*")
    
    # Show prediction probability
    st.write(f"*Prediction Probability*: {prediction_proba[0][1]:.2f} (higher values indicate greater likelihood of diabetes)")

    # Display a progress bar based on probability
    st.progress(int(prediction_proba[0][1] * 100))

    # Visualize the input data as a feature importance chart
    st.subheader("Input Data Breakdown")
    feature_names = [
        "Pregnancies", "Glucose", "Blood Pressure", 
        "Skin Thickness", "Insulin", "BMI", 
        "Diabetes Pedigree Function", "Age"
    ]
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 4))
    input_df.T.plot(kind="barh", ax=ax, legend=False, color="skyblue")
    ax.set_title("Input Features Overview")
    ax.set_xlabel("Value")
    st.pyplot(fig)

    # Visualize the probability with a bar chart
    st.subheader("Prediction Probability Visualization")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["No Diabetes", "Diabetes"], y=prediction_proba[0], ax=ax, palette="Blues")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")
    st.pyplot(fig)

    # Add a feature importance chart (if available from the model)
    st.subheader("Feature Importance (Model Based)")
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="coolwarm")
            ax.set_title("Model Feature Importance")
            st.pyplot(fig)
    except Exception as e:
        st.write("Model doesn't provide feature importances.")
        st.write(f"Error: {e}")
    
        
# Footer
st.markdown("---")
st.markdown("📊 *Diabetes Prediction App* | Powered by Machine Learning, SHAP, and Streamlit")