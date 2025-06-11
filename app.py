import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data to get column types (structure)
@st.cache_data
def load_data():
    df = pd.read_csv('cancer_dataset.csv')
    df.dropna(inplace=True)
    return df

df = load_data()

# Class mapping function
def map_survival_class(months):
    if months < 12:
        return "<1yr"
    elif months < 36:
        return "1-3yrs"
    elif months < 60:
        return "3-5yrs"
    else:
        return ">5yrs"

df['Survival_Class'] = df.iloc[:, -2].apply(map_survival_class)
X_raw = df.iloc[:, :6]
y_raw = df['Survival_Class']

# Identify columns
categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

# Ensure 'Stage' is treated as categorical
if 'Stage' in X_raw.columns and X_raw['Stage'].dtype != object:
    X_raw['Stage'] = X_raw['Stage'].astype(str)
    categorical_cols.append('Stage')
    if 'Stage' in numerical_cols:
        numerical_cols.remove('Stage')

# Preprocessor and Label Encoder
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
X_processed = preprocessor.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_

# Load trained model
model = load_model("cancer_survival_model.h5")

# Streamlit UI
st.title("🧬 Cancer Survival Duration Predictor")
st.markdown("Enter patient data below to predict survival class:")

user_input = {}

# UI label mapping
label_display_names = {
    "gender": "Gender",
    "cancer_type": "Cancer Type",
    "treatment_received": "Treatment Received"
}

stage_mapping = {
    "Stage I": '1',
    "Stage II": '2',
    "Stage III": '3',
    "Stage IV": '4'
}

for col in X_raw.columns:
    col_lower = col.lower()
    display_name = label_display_names.get(col_lower, col.replace('_', ' ').title())

    if col_lower == 'stage':
        stage_label = st.selectbox("Stage", list(stage_mapping.keys()))
        user_input['stage'] = stage_mapping[stage_label]

    elif col_lower == 'tumor_size':
        val = st.number_input("Tumor Size (cm)", min_value=0.0, step=0.1)
        user_input['tumor_size'] = val

    elif col_lower == 'age':
        val = st.number_input("Age", min_value=0, step=1)
        user_input['age'] = int(val)

    elif col in categorical_cols:
        options = sorted(df[col].dropna().unique().tolist())
        val = st.selectbox(display_name, options)
        user_input[col] = val

    else:
        val = st.number_input(display_name, step=1.0)
        user_input[col] = val

# Predict button
if st.button("Predict Survival"):
    input_df = pd.DataFrame([user_input])
    input_transformed = preprocessor.transform(input_df)
    probs = model.predict(input_transformed)[0]
    top_idx = np.argmax(probs)
    top_label = label_encoder.inverse_transform([top_idx])[0]

    st.markdown(f"### 🎯 Most Likely Survival Duration: **{top_label}**")
    st.markdown("#### 📊 Survival Class Probabilities:")
    for idx, prob in enumerate(probs):
        label = label_encoder.inverse_transform([idx])[0]
        st.markdown(f"- **{label}**: `{prob * 100:.1f}%`")
