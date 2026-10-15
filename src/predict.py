
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Locate the project root and saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_risk_xgboost.joblib"
)

# Load the saved model and threshold
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
THRESHOLD = artifact["threshold"]



def predict_credit_risk(customer_data: dict) -> dict:
    """Validate customer data and predict loan default risk."""

    required_fields = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
    ]

    # 1. Validate input type
    if not isinstance(customer_data, dict):
        raise TypeError("Customer data must be a dictionary.")

    # 2. Check required fields
    missing_fields = [
        field for field in required_fields
        if field not in customer_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    # 3. Validate numeric fields
    numeric_fields = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    validated_data = customer_data.copy()

    for field in numeric_fields:
        try:
            value = float(validated_data[field])
        except (TypeError, ValueError):
            raise ValueError(
                f"{field} must be a valid number."
            ) from None

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be finite."
            )

        validated_data[field] = value

    # 4. Validate numerical ranges
    if not 18 <= validated_data["person_age"] <= 100:
        raise ValueError("person_age must be between 18 and 100.")

    if validated_data["person_income"] <= 0:
        raise ValueError("person_income must be greater than zero.")

    if validated_data["person_emp_length"] < 0:
        raise ValueError("person_emp_length cannot be negative.")

    if validated_data["loan_amnt"] <= 0:
        raise ValueError("loan_amnt must be greater than zero.")

    if validated_data["loan_int_rate"] <= 0:
        raise ValueError("loan_int_rate must be greater than zero.")

    if not 0 <= validated_data["loan_percent_income"] <= 1:
        raise ValueError(
            "loan_percent_income must be between 0 and 1."
        )

    if validated_data["cb_person_cred_hist_length"] < 0:
        raise ValueError(
            "Credit history length cannot be negative."
        )

    # 5. Validate categorical fields
    category_options = {
        "person_home_ownership": {
            "RENT", "OWN", "MORTGAGE", "OTHER"
        },
        "loan_intent": {
            "PERSONAL", "EDUCATION", "MEDICAL",
            "VENTURE", "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        },
        "loan_grade": set("ABCDEFG"),
        "cb_person_default_on_file": {"Y", "N"},
    }

    for field, allowed_values in category_options.items():
        value = validated_data[field]

        if not isinstance(value, str):
            raise ValueError(
                f"{field} must be a string."
            )

        value = value.strip().upper()

        if value not in allowed_values:
            raise ValueError(
                f"Invalid {field}: {value}. "
                f"Allowed values: {sorted(allowed_values)}"
            )

        validated_data[field] = value

    # 6. Create DataFrame and engineer the training feature
    customer_df = pd.DataFrame([validated_data])

    customer_df["person_income_log"] = np.log1p(
        customer_df["person_income"]
    )

    # 7. Predict default probability using saved XGBoost model
    probability = model.predict_proba(customer_df)[0, 1]

    # 8. Apply the saved threshold
    prediction = int(probability >= THRESHOLD)

    return {
        "default_probability": round(
            float(probability) * 100, 2
        ),
        "prediction": prediction,
        "risk_class": (
            "Default Risk Detected"
            if prediction == 1
            else "No Default Risk Flag"
        ),
        "threshold": THRESHOLD,
    }
