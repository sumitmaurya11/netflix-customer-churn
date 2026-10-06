# Netflix Customer Churn Prediction and Retention Analysis

## Project Overview

Customer churn is an important business problem for subscription-based platforms. Losing customers can reduce recurring revenue and customer lifetime value.

This project analyzes customer behavior and develops a machine learning system to predict the probability of customer churn.

The project covers data cleaning, exploratory data analysis, feature engineering, machine learning, customer risk segmentation, and retention recommendations.

## Business Objective

The main objectives of this project are:

* Understand customer churn patterns
* Identify customer characteristics associated with churn
* Analyze subscription and engagement behavior
* Build machine learning models to predict churn
* Estimate individual customer churn probability
* Segment customers based on churn risk
* Identify customers who may require retention action
* Translate predictions into business recommendations

## Dataset

The project uses a public Netflix customer churn dataset containing 5,000 customer records and 14 original columns.

The dataset contains information about:

* Customer ID
* Age
* Gender
* Subscription Type
* Monthly Fee
* Watch Hours
* Last Login Days
* Region
* Device
* Payment Method
* Number of Profiles
* Average Watch Time Per Day
* Favorite Genre
* Churn Status

### Target Variable

`churned`

* `0` = Customer stayed
* `1` = Customer churned

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook
* Git
* GitHub

## Project Workflow

```text
Raw Data
    |
    v
Data Cleaning
    |
    v
Exploratory Data Analysis
    |
    v
Feature Engineering
    |
    v
Train/Test Split
    |
    v
Data Preprocessing
    |
    v
Machine Learning
    |
    v
Model Evaluation
    |
    v
Churn Probability
    |
    v
Customer Risk Segmentation
    |
    v
Retention Recommendations
```

## Data Cleaning

The data preparation process includes:

* Removing duplicate records
* Removing duplicate customer IDs
* Converting numerical columns to appropriate data types
* Handling missing numerical values
* Handling missing categorical values
* Validating numerical fields

The cleaned dataset is saved as:

`data/processed/netflix_customer_churn_clean.csv`

## Exploratory Data Analysis

The analysis investigates the relationship between churn and:

* Subscription type
* Gender
* Device
* Payment method
* Favorite genre
* Region
* Age
* Monthly fee
* Watch hours
* Last login activity
* Number of profiles
* Average daily watch time

The project also analyzes customer engagement and inactivity patterns.

## Feature Engineering

Four additional features were created.

### Engagement Score

```text
engagement_score =
watch_hours / (last_login_days + 1)
```

This feature combines viewing activity with recent login behavior.

### Inactivity Level

Customers are classified into:

* Active
* Recently Inactive
* Inactive
* Highly Inactive

### Watch Time Level

Customers are classified into:

* Low
* Medium
* High
* Very High

### Fee Level

Customers are classified into:

* Low
* Medium
* High
* Very High

## Machine Learning

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using stratified 5-fold cross-validation on the training dataset.

The test dataset was kept separate for final model evaluation.

## Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

ROC-AUC was used as the primary model comparison metric.

The final model performance is stored in:

`data/processed/final_model_metrics.csv`

## Customer Churn Prediction

The trained model generates an individual churn probability for each customer.

Example:

```text
Churn Probability: 72.00%
Predicted Churn: Yes
Risk Level: High Risk
```

The prediction system is implemented in:

`src/predict_churn.py`

## Customer Risk Segmentation

Customers are classified into three risk levels:

| Risk Level  | Churn Probability |
| ----------- | ----------------- |
| Low Risk    | Below 30%         |
| Medium Risk | 30% to 60%        |
| High Risk   | Above 60%         |

These risk levels can be used to prioritize customer retention activities.

## Retention Segmentation

The project also creates business-oriented retention segments:

* Win Back Inactive
* Re-engage Low Watch Time
* High Risk Retention
* Medium Risk Monitoring
* Low Risk

## Business Recommendations

### High-Risk Inactive Customers

Customers with high churn probability and long periods since their last login can be targeted with:

* Personalized re-engagement campaigns
* Content recommendations
* Limited-time retention offers

### High-Risk Low-Engagement Customers

Customers with high churn probability and low watch activity can be targeted with:

* Personalized content recommendations
* New-release notifications
* Genre-based recommendations

### Medium-Risk Customers

Medium-risk customers can be monitored through:

* Engagement tracking
* Personalized recommendations
* Targeted communication

## Project Structure

```text
netflix-customer-churn/
|
├── data/
|   ├── raw/
|   |   └── netflix_customer_churn.csv
|   |
|   └── processed/
|       ├── netflix_customer_churn_clean.csv
|       ├── netflix_customer_churn_features.csv
|       ├── customer_churn_predictions.csv
|       ├── high_risk_customers.csv
|       ├── final_model_metrics.csv
|       ├── kpi_summary.csv
|       └── subscription_summary.csv
|
├── models/
|   └── netflix_churn_model.pkl
|
├── notebooks/
|   └── 01_data_understanding.ipynb
|
├── src/
|   ├── data_cleaning.py
|   ├── feature_engineering.py
|   ├── model_training.py
|   └── predict_churn.py
|
├── reports/
|   └── figures/
|
├── app/
|
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project

```bash
cd netflix-customer-churn
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run data cleaning

```bash
python src/data_cleaning.py
```

### 7. Run feature engineering

```bash
python src/feature_engineering.py
```

### 8. Train the machine learning model

```bash
python src/model_training.py
```

### 9. Run customer churn prediction

```bash
python src/predict_churn.py
```

## Key Skills Demonstrated

This project demonstrates practical skills in:

* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Feature Engineering
* Statistical Analysis
* Classification
* Machine Learning
* Cross-Validation
* Model Evaluation
* Customer Segmentation
* Risk Prediction
* Business Analytics
* Python Programming

## Future Improvements

Potential improvements include:

* Hyperparameter tuning
* Advanced ensemble models
* SHAP-based model explainability
* Improved risk thresholds
* Interactive prediction interface
* Model monitoring
* Web application deployment

## Disclaimer

This project uses a public dataset for educational and portfolio purposes. It is not affiliated with or based on proprietary Netflix customer data.
