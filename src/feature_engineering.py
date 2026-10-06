import pandas as pd


def create_features(df):
    """Create features for customer churn analysis."""

    df = df.copy()

    # Engagement score
    df["engagement_score"] = (
        df["watch_hours"] /
        (df["last_login_days"] + 1)
    )

    # Inactivity level
    df["inactivity_level"] = pd.cut(
        df["last_login_days"],
        bins=[-1, 7, 30, 60, float("inf")],
        labels=[
            "Active",
            "Recently Inactive",
            "Inactive",
            "Highly Inactive"
        ]
    )

    # Watch time level
    df["watch_time_level"] = pd.cut(
        df["watch_hours"],
        bins=[-1, 10, 30, 60, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    # Monthly fee level
    df["fee_level"] = pd.cut(
        df["monthly_fee"],
        bins=[-1, 10, 20, 30, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    return df


if __name__ == "__main__":

    input_path = "data/processed/netflix_customer_churn_clean.csv"
    output_path = "data/processed/netflix_customer_churn_features.csv"

    df = pd.read_csv(input_path)

    df_features = create_features(df)

    df_features.to_csv(
        output_path,
        index=False
    )

    print("Feature engineering completed.")
    print("Rows:", len(df_features))
    print("Columns:", len(df_features.columns))