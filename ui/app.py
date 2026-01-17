import streamlit as st
import pandas as pd
import requests

API_ROOT = "http://127.0.0.1:8000"
PREDICT_FILE_URL = f"{API_ROOT}/predict_file"
PREDICT_ROW_URL = f"{API_ROOT}/predict_row"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IVF Trigger Decision Support",
    page_icon="🧬",
    layout="wide",
)

# ---------- CUSTOM CSS THEME ----------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #101827 0, #020617 40%, #020617 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    }
    .main-card {
        background: linear-gradient(135deg, #020617 0%, #020617 40%, #020617 60%, #020617 100%);
        border-radius: 18px;
        padding: 1.5rem 2rem 1.8rem 2rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.9);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg,#38bdf8,#a855f7,#ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.4rem;
    }
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.9rem !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        background-color: rgba(15,23,42,0.85) !important;
        color: #e5e7eb !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg,#38bdf8,#6366f1) !important;
        color: #0b1220 !important;
        border-color: transparent !important;
        box-shadow: 0 10px 20px rgba(56,189,248,0.35);
    }
    .stFileUploader > label {
        font-weight: 600;
        color: #e5e7eb;
    }
    .stButton > button {
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        border: none;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: linear-gradient(90deg,#22c55e,#16a34a);
        color: #0f172a;
        box-shadow: 0 15px 30px rgba(34,197,94,0.35);
    }
    .stButton > button:hover {
        filter: brightness(1.06);
        box-shadow: 0 18px 36px rgba(34,197,94,0.5);
    }
    .result-card {
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        background: radial-gradient(circle at top left,#0f172a,#020617);
        border: 1px solid rgba(148,163,184,0.45);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1.4rem;
        max-width: 1300px;
    }
    .field-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.18rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HEADER ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="app-title">IVF Trigger Decision Support</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload IVF cycle data or enter a single patient to get ML‑driven trigger recommendations in real time.</div>',
    unsafe_allow_html=True,
)

tab_batch, tab_single = st.tabs(
    ["Batch prediction • Excel / CSV", "Single patient • Live input"]
)

# ---------- BATCH TAB ----------
with tab_batch:
    st.markdown("##### Batch prediction from Excel / CSV")

    uploaded_file = st.file_uploader(
        "Upload IVF cycle file (Excel / CSV)",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        st.success(f"Loaded file: {uploaded_file.name}")

        col_run, col_hint = st.columns([1, 2])
        with col_run:
            run_clicked = st.button("Run batch prediction")
        with col_hint:
            st.caption("Data is sent only to your local API at 127.0.0.1.")

        if run_clicked:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(PREDICT_FILE_URL, files=files)

                if response.status_code != 200:
                    st.error(f"API error {response.status_code}: {response.text}")
                else:
                    data = response.json()
                    df_pred = pd.DataFrame(data)

                    st.markdown("###### Predictions")
                    st.dataframe(
                        df_pred,
                        use_container_width=True,
                        height=260,
                    )

                    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions as CSV",
                        data=csv_bytes,
                        file_name="trigger_day_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error calling API: {e}")
    else:
        st.info("Drop an IVF cycle Excel/CSV file above to enable batch prediction.")

# ---------- SINGLE PATIENT TAB ----------
with tab_single:
    st.markdown("##### Single patient prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="field-label">Patient</div>', unsafe_allow_html=True)
        patient_id = st.text_input("Patient ID", "P0001", label_visibility="collapsed")
        age = st.number_input("Age (years)", min_value=18, max_value=55, value=32)
        amh_ng_ml = st.number_input("AMH (ng/mL)", min_value=0.0, value=2.0)
        day = st.number_input("Cycle day", min_value=1, max_value=30, value=8)

    with col2:
        st.markdown('<div class="field-label">Scan metrics</div>', unsafe_allow_html=True)
        avg_follicle_size_mm = st.number_input(
            "Avg follicle size (mm)", min_value=0.0, value=16.0
        )
        follicle_count = st.number_input("Follicle count", min_value=0, value=12)
        estradiol_pg_ml = st.number_input(
            "Estradiol (pg/mL)", min_value=0.0, value=800.0
        )
        progesterone_ng_ml = st.number_input(
            "Progesterone (ng/mL)", min_value=0.0, value=0.7
        )

    with col3:
        st.markdown('<div class="field-label">Derived features</div>', unsafe_allow_html=True)
        age_group = st.selectbox("Age group", ["<35", "35-37", "38-40", ">40"])
        amh_group = st.selectbox("AMH group", ["Low", "Normal", "High"])
        follicle_size_band = st.selectbox(
            "Follicle size band", ["<12", "12-19", ">=20"]
        )
        follicle_size_12_19 = st.number_input(
            "Follicles 12–19mm", min_value=0, value=8
        )
        high_follicle_count = st.selectbox("High follicle count", [0, 1])
        high_e2 = st.selectbox("High E2", [0, 1])
        high_p4 = st.selectbox("High P4", [0, 1])
        late_cycle = st.selectbox("Late cycle", [0, 1])

    st.markdown("---")

    predict_clicked = st.button("Predict trigger for this patient")

    if predict_clicked:
        record = {
            "patient_id": patient_id,
            "age": age,
            "amh_ng_ml": amh_ng_ml,
            "day": day,
            "avg_follicle_size_mm": avg_follicle_size_mm,
            "follicle_count": follicle_count,
            "estradiol_pg_ml": estradiol_pg_ml,
            "progesterone_ng_ml": progesterone_ng_ml,
            "age_group": age_group,
            "amh_group": amh_group,
            "follicle_size_band": follicle_size_band,
            "follicle_size_12_19": follicle_size_12_19,
            "high_follicle_count": high_follicle_count,
            "high_e2": high_e2,
            "high_p4": high_p4,
            "late_cycle": late_cycle,
        }

        try:
            response = requests.post(PREDICT_ROW_URL, json=record)
            if response.status_code != 200:
                st.error(f"API error {response.status_code}: {response.text}")
            else:
                result = response.json()
                pred = result["pred_trigger_recommended"]
                proba = result["pred_trigger_probability"]

                left, right = st.columns(2)

                with left:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="field-label">Trigger decision</div>
                            <div style="font-size: 1.6rem; font-weight: 700;">
                                {"Trigger" if pred == 1 else "Do not trigger"}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with right:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="field-label">Trigger probability</div>
                            <div style="font-size: 1.6rem; font-weight: 700;">
                                {proba:.3f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Error calling API: {e}")

st.markdown("</div>", unsafe_allow_html=True)
