# LOCAL TRAINING PIPELINE WITH OOP AND MLFLOW
import os
import re
import warnings
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# 1. PREPROCESSING CLASS

class DataPreprocessor:
    def __init__(self, target_column="Credit_Score"):
        self.target_column = target_column
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.numerical_cols = None
        self.categorical_cols = None

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

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

    def clean_data(self, df):
        df_clean = df.copy()

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

        # Replace invalid values with NaN
        invalid_values = ["_______", "_", "!@9#%8", "#F%$D@*&8", "NM", "__10000__"]
        df_clean = df_clean.replace(invalid_values, np.nan)

        # Convert numeric columns stored as object
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

        # Convert Month into ordinal number
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

        # Convert Credit_History_Age into total months
        if "Credit_History_Age" in df_clean.columns:
            df_clean["Credit_History_Age_Months"] = df_clean["Credit_History_Age"].apply(
                self.credit_history_to_months
            )
            df_clean = df_clean.drop(columns=["Credit_History_Age"])

        # Feature engineering for Type_of_Loan
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

        # Handle illogical numerical values
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

        print(f"Data cleaning completed. Shape after cleaning: {df_clean.shape}")
        return df_clean

    def count_loan_types(self, value):
        if pd.isna(value) or value == "":
            return 0

        value = str(value).replace("and ", "")
        loans = [loan.strip() for loan in value.split(",") if loan.strip() != ""]
        return len(loans)

    def split_features_target(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        y_encoded = self.label_encoder.fit_transform(y)

        print("Target label mapping:")
        for label, encoded in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)):
            print(f"{label}: {encoded}")

        return X, y_encoded

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape : {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def build_preprocessor(self, X_train):
        self.numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols)
            ],
            remainder="drop"
        )

        print("Preprocessor created successfully.")
        print(f"Numerical columns   : {len(self.numerical_cols)}")
        print(f"Categorical columns : {len(self.categorical_cols)}")

        return self.preprocessor

    def fit_transform(self, X_train, X_test):
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        print("Preprocessing completed.")
        print(f"X_train_processed shape: {X_train_processed.shape}")
        print(f"X_test_processed shape : {X_test_processed.shape}")

        return X_train_processed, X_test_processed

    def save_objects(self, output_dir="artifacts"):
        os.makedirs(output_dir, exist_ok=True)

        joblib.dump(self.preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
        joblib.dump(self.label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

        print("Preprocessor and label encoder saved successfully.")


# 2. Trainning Class

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

        self.trained_models = {}

    def train_model(self, model_name, model, X_train, y_train):
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model


# 3. Evaluation Class

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro")
        }

        return metrics, y_pred

    def print_report(self, y_test, y_pred, label_encoder):
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        ))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


# 4. Pipeline Runner


class TrainingPipeline:
    def __init__(self, data_path, experiment_name="Credit Score Local Training Pipeline"):
        self.data_path = data_path
        self.experiment_name = experiment_name

        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_score = -1

    def run(self):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)

        # Load and clean data
        df = self.preprocessor.load_data(self.data_path)
        df_clean = self.preprocessor.clean_data(df)

        # Split features and target
        X, y = self.preprocessor.split_features_target(df_clean)

        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)

        # Build and apply preprocessing
        self.preprocessor.build_preprocessor(X_train)
        X_train_processed, X_test_processed = self.preprocessor.fit_transform(X_train, X_test)

        # Save preprocessing objects
        self.preprocessor.save_objects()

        # Train and evaluate each model
        for model_name, model in self.trainer.models.items():

            with mlflow.start_run(run_name=model_name):
                trained_model = self.trainer.train_model(
                    model_name,
                    model,
                    X_train_processed,
                    y_train
                )

                metrics, y_pred = self.evaluator.evaluate_model(
                    trained_model,
                    X_test_processed,
                    y_test
                )

                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("target_column", self.preprocessor.target_column)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("num_features", X_train_processed.shape[1])

                # Log model-specific parameters
                if hasattr(trained_model, "get_params"):
                    for param_name, param_value in trained_model.get_params().items():
                        mlflow.log_param(param_name, param_value)

                # Log metrics
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("precision_macro", metrics["precision_macro"])
                mlflow.log_metric("recall_macro", metrics["recall_macro"])
                mlflow.log_metric("f1_macro", metrics["f1_macro"])

                # Log model to MLflow
                mlflow.sklearn.log_model(
                    trained_model,
                    artifact_path="model"
                )

                # Print result
                print("\nEvaluation Result:")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")

                self.evaluator.print_report(
                    y_test,
                    y_pred,
                    self.preprocessor.label_encoder
                )

                # Store result
                result = {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision_Macro": metrics["precision_macro"],
                    "Recall_Macro": metrics["recall_macro"],
                    "F1_Macro": metrics["f1_macro"]
                }

                self.results.append(result)

                # Select best model based on F1 Macro
                if metrics["f1_macro"] > self.best_score:
                    self.best_score = metrics["f1_macro"]
                    self.best_model = trained_model
                    self.best_model_name = model_name

        self.save_results()

    def save_results(self):
        os.makedirs("artifacts", exist_ok=True)

        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values(by="F1_Macro", ascending=False)

        results_df.to_csv("artifacts/model_comparison.csv", index=False)
        joblib.dump(self.best_model, "artifacts/best_model.pkl")

        print("\nTRAINING PIPELINE FINISHED")
        print("Best Model:", self.best_model_name)
        print("Best F1 Macro:", round(self.best_score, 4))
        print("\nModel comparison:")
        print(results_df)

        print("\nSaved files:")
        print("- artifacts/best_model.pkl")
        print("- artifacts/preprocessor.pkl")
        print("- artifacts/label_encoder.pkl")
        print("- artifacts/model_comparison.csv")


if __name__ == "__main__":
    DATA_PATH = "data_C.csv"

    pipeline = TrainingPipeline(
        data_path=DATA_PATH,
        experiment_name="Credit Score Local Training Pipeline"
    )

    pipeline.run()