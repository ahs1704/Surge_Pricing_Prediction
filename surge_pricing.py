import streamlit as st
st.set_page_config(page_title="Sigma Cabs Surge Pricing", layout="wide")

import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 1. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'processed_taxi_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_pipeline.pkl')
ENC_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# -----------------------------------------------------------------------------
# 2. Load Data
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
X = df.drop(['Surge_Pricing_Type', 'Trip_ID'], axis=1)
y = df['Surge_Pricing_Type']

# -----------------------------------------------------------------------------
# 3. Train/Test Split & Label Encode
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# For multiclass metrics
classes = list(range(len(le.classes_)))
y_test_bin = label_binarize(y_test_enc, classes=classes)

# -----------------------------------------------------------------------------
# 4. Preprocessing
numeric_features = [
    'Trip_Distance', 'Customer_Since_Months', 'Life_Style_Index',
    'Customer_Rating', 'Cancellation_Last_1Month', 'Var2', 'Var3'
]
categorical_features = [
    'Type_of_Cab', 'Confidence_Life_Style_Index', 'Destination_Type',
    'Gender', 'Life_Style_Index_Is_Missing', 'Trip_Distance_Category',
    'Customer_Loyalty_Category', 'Customer_Rating_Category',
    'Cancellation_Category', 'CLV_Segment'
]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# -----------------------------------------------------------------------------
# 5. Training & Save Pipeline (if not exist)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hapus model lama agar kompatibel saat pindah versi Python
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
if os.path.exists(ENC_PATH):
    os.remove(ENC_PATH)

if not os.path.exists(MODEL_PATH):
    st.sidebar.info("Training pipeline, please wait...")

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train_enc)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENC_PATH)
    st.sidebar.success("Training complete. Model saved.")
else:
    pipeline = joblib.load(MODEL_PATH)
    le = joblib.load(ENC_PATH)

# -----------------------------------------------------------------------------
# 6. Streamlit App
st.title("🚖 Sigma Cabs Surge Pricing Prediction")
method = st.radio("Input data via:", ("Upload CSV", "Manual Entry"))

if method == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.subheader("Data Preview")
        st.dataframe(df_in)

        preds = pipeline.predict(df_in)
        probs = pipeline.predict_proba(df_in)

        df_out = df_in.copy()
        df_out['Predicted_Surge_Pricing_Type'] = le.inverse_transform(preds)
        proba_df = pd.DataFrame(probs, columns=le.classes_)

        st.subheader("Predictions")
        st.dataframe(pd.concat([df_out, proba_df], axis=1))

else:
    st.subheader("Manual Input")
    with st.form("manual_input_form"):
        td = st.number_input("Trip Distance (km)", 0.0, 100.0, value=5.0)
        csm = st.number_input("Customer Since (months)", 0, 120, value=12)
        lsi = st.slider("Life Style Index", 0.0, 5.0, value=3.0)
        cr = st.slider("Customer Rating", 1.0, 5.0, value=4.0)
        cl1 = st.number_input("Cancellations Last 1 Month", 0, 20, value=0)
        v2 = st.number_input("Var2", value=0.0)
        v3 = st.number_input("Var3", value=0.0)

        toc = st.selectbox("Type of Cab", df['Type_of_Cab'].unique())
        cli = st.selectbox("Confidence Lif
