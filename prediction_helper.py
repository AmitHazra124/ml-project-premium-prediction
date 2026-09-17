from pathlib import Path

import pandas as pd
from joblib import load

ARTIFACTS = Path(__file__).parent / "artifacts"

model_rest = load(ARTIFACTS / "model_rest.joblib")
model_young = load(ARTIFACTS / "model_young.joblib")

scaler_rest = load(ARTIFACTS / "scaler_rest.joblib")
scaler_young = load(ARTIFACTS / "scaler_young.joblib")

# Age at or below this goes to the "young" model, above it to the "rest" model.
YOUNG_AGE_LIMIT = 25

# The exact feature columns the models were trained on, in order. Both models
# agree on this list -- the assertion below keeps it honest if artifacts change.
MODEL_FEATURES = [
    "age",
    "number_of_dependants",
    "income_lakhs",
    "insurance_plan",
    "genetical_risk",
    "normalized_risk_score",
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity",
    "bmi_category_Overweight",
    "bmi_category_Underweight",
    "smoking_status_Occasional",
    "smoking_status_Regular",
    "employment_status_Salaried",
    "employment_status_Self-Employed",
]

for _name, _model in [("young", model_young), ("rest", model_rest)]:
    _trained_on = list(getattr(_model, "feature_names_in_", MODEL_FEATURES))
    assert _trained_on == MODEL_FEATURES, (
        f"model_{_name} was trained on different features than MODEL_FEATURES:\n"
        f"  trained on: {_trained_on}\n"
        f"  expected:   {MODEL_FEATURES}"
    )

# Categories dropped during one-hot encoding (drop_first). They are the implicit
# baseline -- an input of "Female" is simply gender_Male = 0, not its own column.
# The extra smoking labels are data-cleaning artifacts of "No Smoking".
DUMMY_BASELINES = {
    "Female",
    "Northeast",
    "Married",
    "Normal",
    "No Smoking",
    "Smoking=0",
    "Does Not Smoke",
    "Not Smoking",
    "Freelancer",
}

# value -> the dummy column it sets to 1
DUMMY_COLUMNS = {
    "Male": "gender_Male",
    "Northwest": "region_Northwest",
    "Southeast": "region_Southeast",
    "Southwest": "region_Southwest",
    "Unmarried": "marital_status_Unmarried",
    "Obesity": "bmi_category_Obesity",
    "Overweight": "bmi_category_Overweight",
    "Underweight": "bmi_category_Underweight",
    "Occasional": "smoking_status_Occasional",
    "Regular": "smoking_status_Regular",
    "Salaried": "employment_status_Salaried",
    "Self Employed": "employment_status_Self-Employed",
}

CATEGORICAL_FIELDS = [
    "Gender",
    "Region",
    "Marital_Status",
    "BMI_Category",
    "Smoking_Status",
    "Employment_Status",
]

INSURANCE_PLAN_ENCODING = {"Bronze": 1, "Silver": 2, "Gold": 3}

RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}

# diabetes (6) + heart disease (8) is the worst combination in the data.
MAX_RISK_SCORE = 14
MIN_RISK_SCORE = 0


def calculate_risk_score(medical_history):
    diseases = [d.strip() for d in medical_history.lower().split("&")]

    total_risk_score = sum(RISK_SCORES.get(disease, 0) for disease in diseases)

    return (total_risk_score - MIN_RISK_SCORE) / (MAX_RISK_SCORE - MIN_RISK_SCORE)


def handle_scaling(age, df):
    scaler_object = scaler_young if age <= YOUNG_AGE_LIMIT else scaler_rest

    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0.0

    df["income_level"] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop(columns=["income_level"], inplace=True)

    return df


def preprocess_input(input_dict):
    df = pd.DataFrame(0.0, columns=MODEL_FEATURES, index=[0])

    df["age"] = input_dict["Age"]
    df["number_of_dependants"] = input_dict["Number_Of_Dependants"]
    df["income_lakhs"] = input_dict["Income_Lakhs"]
    df["genetical_risk"] = input_dict["Genetical_Risk"]

    df["insurance_plan"] = INSURANCE_PLAN_ENCODING[input_dict["Insurance_Plan"]]
    df["normalized_risk_score"] = calculate_risk_score(input_dict["Medical_History"])

    for field in CATEGORICAL_FIELDS:
        value = input_dict[field]

        if value in DUMMY_BASELINES:
            continue

        if value not in DUMMY_COLUMNS:
            raise ValueError(f"unrecognised {field} value: {value!r}")

        df[DUMMY_COLUMNS[value]] = 1.0

    df = handle_scaling(input_dict["Age"], df)

    return df[MODEL_FEATURES]


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict["Age"] <= YOUNG_AGE_LIMIT:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(round(float(prediction[0])))
