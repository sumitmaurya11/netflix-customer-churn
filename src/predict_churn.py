import pandas as pd
import joblib


# Load trained model
model = joblib.load(
    "models/netflix_churn_model.pkl"
)


print("\nNetflix Customer Churn Prediction")
print("=================================\n")


# Get customer information
age = int(input("Age: "))

gender = input(
    "Gender (Male/Female/Other): "
)

subscription_type = input(
    "Subscription Type (Basic/Standard/Premium): "
)

monthly_fee = float(
    input("Monthly Fee: ")
)

watch_hours = float(
    input("Watch Hours: ")
)

last_login_days = int(
    input("Days Since Last Login: ")
)

region = input(
    "Region: "
)

device = input(
    "Device: "
)

payment_method = input(
    "Payment Method: "
)

number_of_profiles = int(
    input("Number of Profiles: ")
)

avg_watch_time_per_day = float(
    input("Average Watch Time Per Day: ")
)

favorite_genre = input(
    "Favorite Genre: "
)


# Feature engineering
engagement_score = (
    watch_hours /
    (last_login_days + 1)
)


# Inactivity level
if last_login_days <= 7:
    inactivity_level = "Active"

elif last_login_days <= 30:
    inactivity_level = "Recently Inactive"

elif last_login_days <= 60:
    inactivity_level = "Inactive"

else:
    inactivity_level = "Highly Inactive"


# Watch time level
if watch_hours <= 10:
    watch_time_level = "Low"

elif watch_hours <= 30:
    watch_time_level = "Medium"

elif watch_hours <= 60:
    watch_time_level = "High"

else:
    watch_time_level = "Very High"


# Fee level
if monthly_fee <= 10:
    fee_level = "Low"

elif monthly_fee <= 20:
    fee_level = "Medium"

elif monthly_fee <= 30:
    fee_level = "High"

else:
    fee_level = "Very High"


# Create customer dataframe
customer = pd.DataFrame([{

    "age": age,
    "gender": gender,
    "subscription_type": subscription_type,
    "monthly_fee": monthly_fee,
    "watch_hours": watch_hours,
    "last_login_days": last_login_days,
    "region": region,
    "device": device,
    "payment_method": payment_method,
    "number_of_profiles": number_of_profiles,
    "avg_watch_time_per_day": avg_watch_time_per_day,
    "favorite_genre": favorite_genre,

    "engagement_score": engagement_score,

    "inactivity_level": inactivity_level,

    "watch_time_level": watch_time_level,

    "fee_level": fee_level

}])


# Prediction
churn_probability = model.predict_proba(
    customer
)[:, 1][0]

prediction = model.predict(
    customer
)[0]


# Risk level
if churn_probability >= 0.60:

    risk_level = "High Risk"

elif churn_probability >= 0.30:

    risk_level = "Medium Risk"

else:

    risk_level = "Low Risk"


# Retention recommendation
if risk_level == "High Risk":

    recommendation = (
        "Immediate retention action recommended."
    )

elif risk_level == "Medium Risk":

    recommendation = (
        "Monitor customer and send targeted "
        "engagement offers."
    )

else:

    recommendation = (
        "No immediate retention action required."
    )


# Display result
print("\n")
print("========== Prediction Result ==========")

print(
    f"Churn Probability: "
    f"{churn_probability:.2%}"
)

print(
    f"Predicted Churn: "
    f"{'Yes' if prediction == 1 else 'No'}"
)

print(
    f"Risk Level: {risk_level}"
)

print(
    f"Recommendation: {recommendation}"
)

print("=======================================\n")