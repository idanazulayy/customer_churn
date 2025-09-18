import pandas as pd
import numpy as np
import joblib
import json

# --- 1. טעינת כל האובייקטים השמורים ---
model = joblib.load("../../saved_models/telco.pkl")
scaler = joblib.load("telco_scaler.pkl")

with open('telco_columns.json', 'r') as f:
    final_model_columns = json.load(f)

with open('telco_numerical_cols.json', 'r') as f:
    numerical_cols = json.load(f)

print("Model and preprocessing artifacts loaded successfully.")


# --- 2. פונקציית ה-Preprocessing המרכזית ---
def preprocess_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    מקבלת DataFrame עם דאטה גולמי (שורה אחת או יותר)
    ומחזירה DataFrame מוכן לכניסה למודל.
    """
    df = df.copy()

    # --- א. ניקוי והמרות בסיסיות (מתוך clean_object_numeric ו-fe_total_charges) ---
    # ספציפית עבור TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
        df['TotalCharges'].replace('', '0', inplace=True)  # החלפה ב-0 במקום NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # --- ב. Feature Engineering (בדיוק כמו בקוד האימון) ---
    print("Applying feature engineering...")

    # AverageMonthlyCharge
    # טיפול בחלוקה באפס אם tenure הוא 0
    if 'tenure' in df.columns and df['tenure'].iloc[0] == 0:
        df['AverageMonthlyCharge'] = 0
    else:
        df['AverageMonthlyCharge'] = df['TotalCharges'] / df['tenure']

    df['AverageMonthlyCharge'].replace([np.inf, -np.inf], 0, inplace=True)
    df['AverageMonthlyCharge'].fillna(0, inplace=True)

    # tenure_group
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, np.inf],
                                labels=['0-12', '13-24', '25-48', '49+'],
                                right=False)

    # HasInternetService
    df['HasInternetService'] = df['InternetService'].apply(
        lambda x: 1 if x in ['DSL', 'Fiber optic'] else 0
    )

    # Num_Extra_Services
    extra_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies']
    df['Num_Extra_Services'] = df[extra_services].apply(
        lambda row: sum(1 for service in row if service == 'Yes'), axis=1
    )

    # SeniorCitizen_x_Contract
    df['SeniorCitizen_x_Contract'] = df.apply(
        lambda row: f"{row['SeniorCitizen']}_{row['Contract']}", axis=1
    )

    # יצירת bins עבור TotalCharges (זו לא הייתה בפונקציה שלך, אבל היא קריטית)
    # הערה: יצירת bins על דאטה חדש היא מורכבת. גישה פשוטה יותר היא לוותר עליה בפרודקשיין
    # או להשתמש ב-binning שלמדת באימון. לצורך הפשטות, נוותר על הפיצ'ר הזה כרגע.
    # אם הוא חשוב, צריך לשמור גם את גבולות ה-bins.
    # df = fe_total_charges(df, col='TotalCharges', n_bins=15) # זו פעולה מורכבת לפרודקשיין

    # --- ג. טיפול בעמודות לא רלוונטיות ---
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # --- ד. One-Hot Encoding (get_dummies) ---
    df = pd.get_dummies(df)

    # --- ה. התאמת עמודות לפורמט של המודל ---
    # הוספת עמודות חסרות (שלא הופיעו בדאטה החדש אבל היו באימון) עם ערך 0
    # והסרת עמודות עודפות (שהופיעו בדאטה החדש אבל לא היו באימון)
    processed_df = df.reindex(columns=final_model_columns, fill_value=0)

    # --- ו. Scaling ---
    # חשוב: רק transform, לא fit_transform!
    # ודא שהעמודות המספריות קיימות לפני ה-scaling
    cols_to_scale = [col for col in numerical_cols if col in processed_df.columns]
    if cols_to_scale:
        processed_df[cols_to_scale] = scaler.transform(processed_df[cols_to_scale])

    return processed_df


# --- 3. פונקציית חיזוי ---
def predict_churn(customer_data: dict) -> dict:
    """
    מקבלת מילון עם נתוני לקוח, ומחזירה את החיזוי.
    """
    # המרת המילון ל-DataFrame
    df = pd.DataFrame([customer_data])

    # עיבוד הדאטה
    processed_df = preprocess_new_data(df)

    # ביצוע חיזוי (הסתברות לנטישה)
    prediction_proba = model.predict_proba(processed_df)[:, 1]  # [:, 1] נותן את ההסתברות למחלקה "1" (Churn)

    # חיזוי בינארי (0 או 1)
    prediction = model.predict(processed_df)

    return {
        'prediction': int(prediction[0]),
        'churn_probability': float(prediction_proba[0])
    }


# --- דוגמת שימוש ---
if __name__ == '__main__':
    # נתוני לקוח חדש לדוגמה (בפורמט גולמי)
    new_customer = [
        {'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
         'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service', 'InternetService': 'DSL',
         'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No',
         'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': '29.85'},
        {'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 34, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
         'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'No', 'StreamingTV': 'No',
         'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Mailed check',
         'MonthlyCharges': 56.95, 'TotalCharges': '1889.5'},
        {'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
         'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
         'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Mailed check', 'MonthlyCharges': 53.85, 'TotalCharges': '108.15'},
        {'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 45, 'PhoneService': 'No', 'MultipleLines': 'No phone service', 'InternetService': 'DSL',
         'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
         'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'No',
         'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 42.3, 'TotalCharges': '1840.75'},
        {'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
         'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
         'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.7, 'TotalCharges': '151.65'},
        {'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 8, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
         'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'No',
         'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Electronic check', 'MonthlyCharges': 99.65, 'TotalCharges': '820.5'},
        {'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'Yes',
         'tenure': 22, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
         'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No',
         'StreamingTV': 'Yes', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Credit card (automatic)', 'MonthlyCharges': 89.1, 'TotalCharges': '1949.4'},
        {'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
         'tenure': 10, 'PhoneService': 'No', 'MultipleLines': 'No phone service', 'InternetService': 'DSL',
         'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
         'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'No',
         'PaymentMethod': 'Mailed check', 'MonthlyCharges': 29.75, 'TotalCharges': '301.9'},
        {'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
         'tenure': 28, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'Fiber optic',
         'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
         'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
         'PaymentMethod': 'Electronic check', 'MonthlyCharges': 104.8, 'TotalCharges': '3046.05'},
        {'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'Yes',
         'tenure': 62, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
         'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
         'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'No',
         'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 56.15, 'TotalCharges': '3487.95',
         }
    ]

for data in new_customer:
    result = predict_churn(data)
    print(f"\nPrediction for new customer:")
    print(f"Churn Prediction (0=No, 1=Yes): {result['prediction']}")
    print(f"Probability of Churn: {result['churn_probability']:.2%}")
