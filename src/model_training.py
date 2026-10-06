import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

input_path = "data/processed/netflix_customer_churn_features.csv"

df = pd.read_csv(input_path)

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Prepare features and target
# --------------------------------------------------

model_df = df.drop(
    columns=["customer_id"]
).copy()

categorical_features = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
    "inactivity_level",
    "watch_time_level",
    "fee_level"
]

for col in categorical_features:
    model_df[col] = model_df[col].astype(str)

X = model_df.drop(columns=["churned"])
y = model_df["churned"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_features = [
    "age",
    "monthly_fee",
    "watch_hours",
    "last_login_days",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "engagement_score"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )
}


# --------------------------------------------------
# 6. Cross-validation on training data only
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = []


for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({

        "Model": name,

        "Accuracy": scores[
            "test_accuracy"
        ].mean(),

        "Precision": scores[
            "test_precision"
        ].mean(),

        "Recall": scores[
            "test_recall"
        ].mean(),

        "F1 Score": scores[
            "test_f1"
        ].mean(),

        "ROC-AUC": scores[
            "test_roc_auc"
        ].mean()
    })


cv_results_df = pd.DataFrame(cv_results)

print("\nCross-validation results:")
print(
    cv_results_df
    .round(3)
    .sort_values(
        "ROC-AUC",
        ascending=False
    )
)


# --------------------------------------------------
# 7. Select best model
# --------------------------------------------------

best_model_name = (
    cv_results_df
    .sort_values(
        "ROC-AUC",
        ascending=False
    )
    .iloc[0]["Model"]
)

print(
    "\nBest model:",
    best_model_name
)


# --------------------------------------------------
# 8. Train best model
# --------------------------------------------------

final_model = models[best_model_name]

final_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 9. Final test evaluation
# --------------------------------------------------

final_pred = final_model.predict(X_test)

final_proba = final_model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    final_pred
)

precision = precision_score(
    y_test,
    final_pred
)

recall = recall_score(
    y_test,
    final_pred
)

f1 = f1_score(
    y_test,
    final_pred
)

roc_auc = roc_auc_score(
    y_test,
    final_proba
)


print("\nFinal Test Performance:")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")
print(f"ROC-AUC  : {roc_auc:.3f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_pred,
        target_names=[
            "Stayed",
            "Churned"
        ]
    )
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

model_path = "models/netflix_churn_model.pkl"

joblib.dump(
    final_model,
    model_path
)

print(
    "\nModel saved to:",
    model_path
)


# --------------------------------------------------
# 11. Save metrics
# --------------------------------------------------

final_metrics = pd.DataFrame([{

    "model": best_model_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc

}])

metrics_path = (
    "data/processed/"
    "final_model_metrics.csv"
)

final_metrics.to_csv(
    metrics_path,
    index=False
)

print(
    "Metrics saved to:",
    metrics_path
)