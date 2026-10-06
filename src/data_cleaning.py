import pandas as pd


def load_data(file_path):
    """Load the raw Netflix customer churn dataset."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and validate the customer churn dataset."""

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate customer IDs
    df = df.drop_duplicates(
        subset="customer_id",
        keep="first"
    )

    # Convert numerical columns
    numeric_cols = [
        "age",
        "monthly_fee",
        "watch_hours",
        "last_login_days",
        "number_of_profiles",
        "avg_watch_time_per_day",
        "churned"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Handle missing numerical values
    numerical_features = [
        "age",
        "monthly_fee",
        "watch_hours",
        "last_login_days",
        "number_of_profiles",
        "avg_watch_time_per_day"
    ]

    df[numerical_features] = df[numerical_features].fillna(
        df[numerical_features].median()
    )

    # Handle missing categorical values
    categorical_features = [
        "gender",
        "subscription_type",
        "region",
        "device",
        "payment_method",
        "favorite_genre"
    ]

    df[categorical_features] = df[categorical_features].fillna(
        df[categorical_features].mode().iloc[0]
    )

    return df


if __name__ == "__main__":

    input_path = "data/raw/netflix_customer_churn.csv"
    output_path = "data/processed/netflix_customer_churn_clean.csv"

    df = load_data(input_path)
    df_clean = clean_data(df)

    df_clean.to_csv(
        output_path,
        index=False
    )

    print("Data cleaning completed.")
    print("Rows:", len(df_clean))
    print("Columns:", len(df_clean.columns))