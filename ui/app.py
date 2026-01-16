import streamlit as st
import pandas as pd
import requests
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IVF Trigger Decision Support",
    page_icon="🧬",
    layout="wide",
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="text-align:center; color:#1f4e79;">
        IVF Ovarian Stimulation – Trigger Decision Support
    </h1>
    <p style="text-align:center; color:gray;">
        Upload patient cycle data or enter details to get trigger recommendations powered by MLflow models.
    </p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📁 Batch (Excel/CSV)", "👤 Single Patient"])

# ---------- TAB 1: FILE UPLOAD ----------
with tab1:
    st.subheader("Batch prediction from Excel / CSV")

    uploaded_file = st.file_uploader(
        "Upload IVF cycle file (Excel or CSV)",
        type=["csv", "xlsx", "xls"],
        help="File must have the same columns as your preprocessed dataset.",
    )

    if uploaded_file is not None:
        st.success(f"Loaded file: {uploaded_file.name}")

        if st.button("Run batch prediction", type="primary"):
            with st.spinner("Sending data to model API and generating predictions..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_URL}/predict_file", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    df_pred = pd.DataFrame(data)

                    st.success("Predictions generated successfully.")
                    st.dataframe(
                        df_pred.head(20),
                        use_container_width=True,
                    )

                    # Download button
                    csv_buf = io.StringIO()
                    df_pred.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="⬇️ Download predictions as CSV",
                        data=csv_buf.getvalue(),
                        file_name="ivf_trigger_predictions_ui.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")

# ---------- TAB 2: SINGLE PATIENT ----------
with tab2:
    st.subheader("Single patient cycle prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_id = st.text_input("Patient ID", "P0001")
        age = st.number_input("Age (years)", 20, 50, 32)
        amh_ng_ml = st.number_input("AMH (ng/mL)", 0.1, 10.0, 2.5, step=0.1)
        day = st.number_input("Cycle day", 1, 20, 8)

    with col2:
        avg_follicle_size_mm = st.number_input("Avg follicle size (mm)", 5.0, 30.0, 16.0, step=0.1)
        follicle_count = st.number_input("Follicle count", 0, 60, 10)
        estradiol_pg_ml = st.number_input("Estradiol (pg/mL)", 0.0, 5000.0, 800.0, step=10.0)
        progesterone_ng_ml = st.number_input("Progesterone (ng/mL)", 0.0, 10.0, 0.6, step=0.05)

    with col3:
        age_group = st.selectbox("Age group", ["<30", "30-34", "35-37", "38-40", ">40"])
        amh_group = st.selectbox("AMH group", ["low", "normal", "high"])
        follicle_size_band = st.selectbox("Follicle size band", ["<12", "12-19", ">=20"])
        follicle_size_12_19 = st.selectbox("Has 12–19 mm follicles?", [0, 1])
        high_follicle_count = st.selectbox("High follicle count?", [0, 1])
        high_e2 = st.selectbox("High E2?", [0, 1])
        high_p4 = st.selectbox("High P4?", [0, 1])
        late_cycle = st.selectbox("Late cycle?", [0, 1])

    if st.button("Predict trigger decision", type="primary"):
        payload = {
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
            "follicle_size_12_19": int(follicle_size_12_19),
            "high_follicle_count": int(high_follicle_count),
            "high_e2": int(high_e2),
            "high_p4": int(high_p4),
            "late_cycle": int(late_cycle),
        }

        with st.spinner("Calling model API..."):
            resp = requests.post(f"{API_URL}/predict_row", json=payload)
            if resp.status_code == 200:
                out = resp.json()
                pred = out["pred_trigger_recommended"]
                prob = out["pred_trigger_probability"]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Trigger recommended?",
                        "Yes" if pred == 1 else "No",
                    )
                with col_b:
                    st.metric(
                        "Trigger probability",
                        f"{prob*100:.1f} %",
                    )

                st.info("Model: GradientBoosting (logged in MLflow)")
            else:
                st.error(f"API error: {resp.status_code} - {resp.text}")
