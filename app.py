"""Streamlit interface for the glass-type classifier.
Run: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st


# Get the folder where this app.py file is located
HERE = os.path.dirname(os.path.abspath(__file__))


# Glass type descriptions
FULL = {
    "WinF": "Window glass (float processed)",
    "WinNF": "Window glass (non-float)",
    "Veh": "Vehicle window glass",
    "Con": "Container glass",
    "Tabl": "Tableware glass",
    "Head": "Headlamp glass"
}


# Load the trained model and dataset
@st.cache_resource
def load():
    bundle = joblib.load(
        os.path.join(HERE, "glass_model.joblib")
    )

    data = pd.read_csv(
        os.path.join(HERE, "glass.csv")
    ).drop(columns=["rownames"])

    return bundle, data


# Load model
bundle, data = load()

model = bundle["model"]
le = bundle["label_encoder"]
features = bundle["features"]


# Page title
st.title("Forensic Glass Type Classifier")


# Description
st.write(
    "Enter the refractive index and chemical composition "
    "(weight % oxide) of a glass fragment to predict its type. "
    "Defaults are the dataset medians."
)


# Information about RI
st.caption(
    "RI is stored on the dataset's shifted scale: "
    "(refractive index − 1.518) × 1000."
)


# Create three columns for input fields
cols = st.columns(3)

values = {}


# Create input boxes for every feature
for i, f in enumerate(features):

    lo = float(data[f].min())
    hi = float(data[f].max())
    med = float(data[f].median())

    values[f] = cols[i % 3].number_input(
        f,
        min_value=lo - 1,
        max_value=hi + 1,
        value=med,
        step=(hi - lo) / 100
    )


# Prediction button
if st.button("Predict", type="primary"):

    # Convert user input into DataFrame
    x = pd.DataFrame([values])[features]

    # Get prediction probabilities
    proba = model.predict_proba(x)[0]

    # Find class with highest probability
    best = proba.argmax()

    # Get class code
    code = le.classes_[best]

    # Display prediction
    st.success(
        f"Predicted type: **{FULL[code]}** "
        f"(confidence {proba[best]:.0%})"
    )

    # Display probability chart
    probability_data = pd.Series(
        proba,
        index=[FULL[c] for c in le.classes_],
        name="Probability"
    )

    st.bar_chart(probability_data)