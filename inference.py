import re
import joblib
import pandas as pd
import numpy as np


class CreditScoreInference:
    def __init__(
        self,
        model_path="artifacts/best_model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl",
        label_encoder_path="artifacts/label_encoder.pkl"
    ):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def clean_numeric_column(self, series):
        return pd.to_numeric(
            series.astype(str)
                  .str.replace("_", "", regex=False)
                  .str.strip(),
            errors="coerce"
        )

    def credit_history_to_months(self, value):
        if pd.isna(value):
            return np.nan

        value = str(value)

        years = re.search(r"(\d+)\s*Years?", value)
        months = re.search(r"(\d+)\s*Months?", value)

        total_months = 0

        if years:
            total_months += int(years.group(1)) * 12

        if months:
            total_months += int(months.group(1))

        return total_months

    def count_loan_types(self, value):
        if pd.isna(value) or value == "":
            return 0

        value = str(value).replace("and ", "")
        loans = [loan.strip() for loan in value.split(",") if loan.strip() != ""]
        return len(loans)

    def clean_data(self, df):
        df_clean = df.copy()

        # Drop target if exists
        if "Credit_Score" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Credit_Score"])

        # Drop irrelevant identity columns
        drop_columns = [
            "Unnamed: 0",
            "ID",
            "Customer_ID",
            "Name",
            "SSN"
        ]

        df_clean = df_clean.drop(
            columns=[col for col in drop_columns if col in df_clean.columns],
            errors="ignore"
        )

        # Replace invalid values
        invalid_values = ["_______", "_", "!@9#%8", "#F%$D@*&8", "NM", "__10000__"]
        df_clean = df_clean.replace(invalid_values, np.nan)

        # Convert numeric object columns
        numeric_object_cols = [
            "Age",
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly"
        ]

        for col in numeric_object_cols:
            if col in df_clean.columns:
                df_clean[col] = self.clean_numeric_column(df_clean[col])

        # Convert Month
        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        if "Month" in df_clean.columns:
            df_clean["Month"] = df_clean["Month"].map(month_mapping)

        # Convert 
        if "Credit_History_Age" in df_clean.columns:
            df_clean["Credit_History_Age_Months"] = df_clean["Credit_History_Age"].apply(
                self.credit_history_to_months
            )
            df_clean = df_clean.drop(columns=["Credit_History_Age"])

        # Feature engineering Type_of_Loan
        if "Type_of_Loan" in df_clean.columns:
            type_of_loan_raw = df_clean["Type_of_Loan"].fillna("")

            loan_types = [
                "Auto Loan",
                "Credit-Builder Loan",
                "Debt Consolidation Loan",
                "Home Equity Loan",
                "Mortgage Loan",
                "Not Specified",
                "Payday Loan",
                "Personal Loan",
                "Student Loan"
            ]

            for loan in loan_types:
                new_col = "Loan_" + loan.replace(" ", "_").replace("-", "_")
                df_clean[new_col] = type_of_loan_raw.str.contains(
                    loan,
                    case=False,
                    regex=False
                ).astype(int)

            df_clean["Loan_Type_Count"] = type_of_loan_raw.apply(self.count_loan_types)
            df_clean = df_clean.drop(columns=["Type_of_Loan"])

        # illogical values
        value_bounds = {
            "Age": (18, 100),
            "Annual_Income": (0, 500000),
            "Monthly_Inhand_Salary": (0, 50000),
            "Num_Bank_Accounts": (0, 20),
            "Num_Credit_Card": (0, 20),
            "Interest_Rate": (0, 50),
            "Num_of_Loan": (0, 20),
            "Delay_from_due_date": (0, 90),
            "Num_of_Delayed_Payment": (0, 60),
            "Changed_Credit_Limit": (-50, 50),
            "Num_Credit_Inquiries": (0, 50),
            "Outstanding_Debt": (0, 100000),
            "Credit_Utilization_Ratio": (0, 100),
            "Total_EMI_per_month": (0, 50000),
            "Amount_invested_monthly": (0, 50000),
            "Monthly_Balance": (0, 50000),
            "Credit_History_Age_Months": (0, 600)
        }

        for col, (lower, upper) in value_bounds.items():
            if col in df_clean.columns:
                df_clean.loc[
                    (df_clean[col] < lower) | (df_clean[col] > upper),
                    col
                ] = np.nan

        return df_clean

    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        cleaned_data = self.clean_data(input_df)
        processed_data = self.preprocessor.transform(cleaned_data)

        prediction_encoded = self.model.predict(processed_data)
        prediction_label = self.label_encoder.inverse_transform(prediction_encoded)

        return prediction_label

    def predict_with_result(self, input_data):
        input_df = input_data.copy()
        predictions = self.predict(input_df)

        result_df = input_df.copy()
        result_df["Predicted_Credit_Score"] = predictions

        return result_df


if __name__ == "__main__":
    inference = CreditScoreInference()

    sample_data = pd.read_csv("data_C.csv").head(5)
    result = inference.predict_with_result(sample_data)

    print(result[["Predicted_Credit_Score"]])