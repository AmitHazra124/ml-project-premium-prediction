from prediction_helper import predict
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏥 Health Insurance Cost Predictor")
st.write("Enter the customer information below.")

# --------------------------------------------------
# Options
# --------------------------------------------------

gender_options = [
    "Male",
    "Female"
]

region_options = [
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest"
]

marital_status_options = [
    "Unmarried",
    "Married"
]

bmi_category_options = [
    "Overweight",
    "Underweight",
    "Normal",
    "Obesity"
]

smoking_status_options = [
    "No Smoking",
    "Regular",
    "Occasional"
]

employment_status_options = [
    "Self Employed",
    "Freelancer",
    "Salaried"
]

medical_history_options = [
    "No Disease",
    "High blood pressure",
    "Diabetes & High blood pressure",
    "Diabetes & Heart disease",
    "Diabetes",
    "Diabetes & Thyroid",
    "Heart disease",
    "Thyroid",
    "High blood pressure & Heart disease"
]

insurance_plan_options = [
    "Silver",
    "Bronze",
    "Gold"
]

# --------------------------------------------------
# Form
# --------------------------------------------------

with st.form("health_prediction_form"):

    st.subheader("Customer Information")

    # ==================================================
    # Row 1
    # Age | Number of Dependants | Income in Lakhs
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1
        )

    with col2:
        number_of_dependants = st.number_input(
            "Number of Dependants",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    with col3:
        income_lakhs = st.number_input(
            "Income in Lakhs",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )

    # ==================================================
    # Row 2
    # Genetical Risk | Insurance Plan | Employment Status
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        genetical_risk = st.number_input(
            "Genetical Risk",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with col2:
        insurance_plan = st.selectbox(
            "Insurance Plan",
            insurance_plan_options
        )

    with col3:
        employment_status = st.selectbox(
            "Employment Status",
            employment_status_options
        )

    # ==================================================
    # Row 3
    # Gender | Marital Status | BMI Category
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            gender_options
        )

    with col2:
        marital_status = st.selectbox(
            "Marital Status",
            marital_status_options
        )

    with col3:
        bmi_category = st.selectbox(
            "BMI Category",
            bmi_category_options
        )

    # ==================================================
    # Row 4
    # Smoking Status | Region | Medical History
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        smoking_status = st.selectbox(
            "Smoking Status",
            smoking_status_options
        )

    with col2:
        region = st.selectbox(
            "Region",
            region_options
        )

    with col3:
        medical_history = st.selectbox(
            "Medical History",
            medical_history_options
        )

    # ==================================================
    # Predict Button
    # ==================================================

    st.write("")

    submitted = st.form_submit_button(
        "🔮 Predict",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    # Create DataFrame using the EXACT
    # column names from your dataset.

    input_data = {
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Marital_Status": marital_status,
        "Number_Of_Dependants": number_of_dependants,
        "BMI_Category": bmi_category,
        "Smoking_Status": smoking_status,
        "Employment_Status": employment_status,
        "Income_Lakhs": income_lakhs,
        "Medical_History": medical_history,
        "Insurance_Plan": insurance_plan,
        "Genetical_Risk": genetical_risk
    }

    prediction = predict(input_data)
    st.success(f"Predicted Premium {prediction}")
