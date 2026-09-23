import streamlit as st
import pandas as pd
from inference import CreditScoreInference


st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)


# CSS

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #b8b8b8;
        margin-bottom: 30px;
    }

    .section-card {
        padding: 22px;
        border-radius: 16px;
        background-color: #161b22;
        border: 1px solid #30363d;
        margin-bottom: 22px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 14px;
        background-color: #0f172a;
        border: 1px solid #334155;
        text-align: center;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }

    .result-good {
        padding: 20px;
        border-radius: 14px;
        background-color: #123c2c;
        border: 1px solid #22c55e;
        color: #bbf7d0;
        font-size: 22px;
        font-weight: 700;
    }

    .result-standard {
        padding: 20px;
        border-radius: 14px;
        background-color: #1e3a5f;
        border: 1px solid #38bdf8;
        color: #bae6fd;
        font-size: 22px;
        font-weight: 700;
    }

    .result-poor {
        padding: 20px;
        border-radius: 14px;
        background-color: #4a1d1d;
        border: 1px solid #ef4444;
        color: #fecaca;
        font-size: 22px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_inference_model():
    return CreditScoreInference()


def get_test_cases_by_prediction(inference, data_path="data_C.csv"):
    df = pd.read_csv(data_path)
    result_df = inference.predict_with_result(df)

    test_cases = {}

    for label in ["Good", "Standard", "Poor"]:
        filtered = result_df[result_df["Predicted_Credit_Score"] == label]

        if len(filtered) > 0:
            test_cases[label] = filtered.iloc[[0]]

    return test_cases


def show_prediction_result(prediction):
    if prediction == "Good":
        st.markdown(
            f"""
            <div class="result-good">
                Predicted Credit Score: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif prediction == "Standard":
        st.markdown(
            f"""
            <div class="result-standard">
                Predicted Credit Score: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="result-poor">
                Predicted Credit Score: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )



inference = load_inference_model()


# SIDEBAR
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose prediction mode:",
    ["Built-in Test Case", "Upload CSV"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses the trained Random Forest model to predict customer credit score."
)


# HEADER
st.markdown('<div class="main-title">Credit Score Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning deployment app for predicting customer credit score based on financial behavior.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Best Model</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">0.7388</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">F1 Macro</div>
            <div class="metric-value">0.7170</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# TEST CASE

if menu == "Built-in Test Case":
    st.markdown("## Built-in Test Case Prediction")

    st.write(
        "This section uses sample data from the dataset to test the deployed model. "
        "The test cases are selected to represent each predicted credit score class."
    )

    test_cases = get_test_cases_by_prediction(inference)
    available_labels = list(test_cases.keys())

    selected_label = st.selectbox(
        "Choose test case class:",
        available_labels
    )

    selected_data = test_cases[selected_label]
    prediction = selected_data["Predicted_Credit_Score"].iloc[0]

    st.markdown("### Prediction Result")
    show_prediction_result(prediction)

    if "Credit_Score" in selected_data.columns:
        st.caption("Actual label is shown only for checking because the sample comes from the original dataset.")
        st.write("Actual Credit Score:", selected_data["Credit_Score"].iloc[0])

    st.markdown("### Customer Summary")

    summary_cols = [
        "Age",
        "Occupation",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Payment_of_Min_Amount",
        "Payment_Behaviour"
    ]

    summary_cols = [col for col in summary_cols if col in selected_data.columns]

    st.dataframe(
        selected_data[summary_cols],
        use_container_width=True
    )

    with st.expander("Show full input data"):
        st.dataframe(
            selected_data.drop(columns=["Predicted_Credit_Score"], errors="ignore"),
            use_container_width=True
        )



# CSV  
elif menu == "Upload CSV":
    st.markdown("## Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with the same structure as the training dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.markdown("### Uploaded Data Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("Predict Credit Score"):
            result_df = inference.predict_with_result(input_df)

            st.markdown("### Prediction Result")
            st.dataframe(result_df, use_container_width=True)

            csv_result = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Result",
                data=csv_result,
                file_name="credit_score_prediction_result.csv",
                mime="text/csv"
            )