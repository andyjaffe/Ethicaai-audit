import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

st.set_page_config(page_title="EthicaAI - Bias Detection Tool", layout="wide")

st.title("🤖 EthicaAI Bias Audit Tool")
st.write("Upload your dataset and we'll check it for demographic bias in your model's decisions.")

uploaded_file = st.file_uploader("📂 Upload your xlsx or xls file", type=["csv"])

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
import pandas as pd
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)


    label_col = st.selectbox("✅ Choose your target column (e.g. hired, approved)", df.columns)
    score_col = st.selectbox("📈 Choose your model score/confidence column (if any)", df.columns)
    sensitive_col = st.selectbox("⚖️ Choose your sensitive attribute (e.g. gender, race)", df.columns)

    if st.button("🚦 Run Bias Audit"):
        try:
            df_clean = df[[score_col, sensitive_col, label_col]].dropna()
            df_clean[sensitive_col] = LabelEncoder().fit_transform(df_clean[sensitive_col])

            X = df_clean[[score_col, sensitive_col]]
            y = df_clean[label_col]
            sensitive = df[sensitive_col]

            model = LogisticRegression().fit(X, y)
            y_pred = model.predict(X)

            metric_frame = MetricFrame(metrics=selection_rate, y_true=y, y_pred=y_pred, sensitive_features=sensitive)
            dp_diff = demographic_parity_difference(y_true=y, y_pred=y_pred, sensitive_features=sensitive)

            st.subheader("📊 Selection Rates by Group")
            st.write(metric_frame.by_group)

            st.subheader("⚖️ Demographic Parity Difference")
            st.write(f"**{dp_diff:.3f}**")

            if abs(dp_diff) > 0.2:
                st.error("🚨 Significant bias detected! Your model may need fairness adjustments.")
            else:
                st.success("✅ Your model appears fair across the selected demographic — good job!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
else:
    st.info("⬆️ Upload a CSV file to get started.")

