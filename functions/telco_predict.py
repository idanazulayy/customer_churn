import joblib
import pandas as pd
import json
import numpy as np

# --- הגדרת שמות הקבצים ---
MODEL_FILE_NAME = "../saved_models/telco.pkl"
SCALER_FILE_NAME = "../preprocessing/telco/telco_scaler.pkl"
NUMERICAL_COLS_FILE = "../preprocessing/telco/telco_numerical_cols.json"
FINAL_COLS_FILE = "../preprocessing/telco/telco_columns.json"


def fe_total_charges(df, col='TotalCharges', n_bins=15):
    """
    מנקה את העמודה TotalCharges, ממיר לכל float, ממלא ערכים חסרים ב-0.
    """
    df = df.copy()
    df[col] = df[col].replace(" ", np.nan)
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(0)
    return df


def predict_telco_churn(new_data: pd.DataFrame) -> tuple:
    """
    מבצע חיזוי נטישה על נתוני לקוחות Telco חדשים, כולל כל שלבי ה-preprocessing.

    Parameters:
    ----------
    new_data : pd.DataFrame
        DataFrame המכיל את נתוני הלקוחות החדשים.

    Returns:
    -------
    tuple (prediction, probability)
        prediction: 0 (לא נוטש) או 1 (נוטש).
        probability: ההסתברות לנטישה (ערך בין 0 ל-1).
    """

    # 1. טעינת המודל וה-preprocessors מהקבצים
    try:
        telco_model = joblib.load(MODEL_FILE_NAME)
        telco_scaler = joblib.load(SCALER_FILE_NAME)
        with open(NUMERICAL_COLS_FILE, 'r') as f:
            numerical_cols_to_scale = json.load(f)
        with open(FINAL_COLS_FILE, 'r') as f:
            final_columns = json.load(f)
        print(">> מודל Telco וקבצי Preprocessing נטענו בהצלחה.")
    except FileNotFoundError as e:
        print(f"שגיאה: קובץ לא נמצא - {e}. ודא שהקבצים נמצאים באותו נתיב.")
        return None, None

    # 2. ביצוע Feature Engineering על הנתונים החדשים
    print(">> מבצע Feature Engineering על הנתונים החדשים...")
    df_proc = new_data.copy()

    # ניקוי עמודת TotalCharges והמרתה למספרים
    df_proc = fe_total_charges(df_proc)

    # יצירת פיצ'ר AverageMonthlyCharge
    df_proc['AverageMonthlyCharge'] = df_proc['TotalCharges'] / df_proc['tenure']
    df_proc['AverageMonthlyCharge'].replace([np.inf, -np.inf], 0, inplace=True)
    df_proc['AverageMonthlyCharge'].fillna(0, inplace=True)

    # קיטוע tenure לקטגוריות
    df_proc['tenure_group'] = pd.cut(df_proc['tenure'],
                                     bins=[0, 12, 24, 48, np.inf],
                                     labels=['0-12', '13-24', '25-48', '49+'],
                                     right=False)

    # יצירת פיצ'ר HasInternetService
    df_proc['HasInternetService'] = df_proc['InternetService'].apply(
        lambda x: 1 if x in ['DSL', 'Fiber optic'] else 0
    )

    # יצירת פיצ'ר Num_Extra_Services
    extra_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies']
    df_proc['Num_Extra_Services'] = df_proc[extra_services].apply(
        lambda row: sum(1 for service in row if service == 'Yes'), axis=1
    )

    # יצירת אינטראקציה בין SeniorCitizen ו-Contract
    df_proc['SeniorCitizen_x_Contract'] = df_proc.apply(
        lambda row: f"{row['SeniorCitizen']}_{row['Contract']}", axis=1
    )

    # 3. הפרדה של פיצ'רים מספריים וקטגוריאליים
    num_cols_proc = numerical_cols_to_scale
    cat_cols_proc = [col for col in df_proc.columns if
                     col not in num_cols_proc and df_proc.dtypes[col] == 'object' or df_proc.dtypes[
                         col].name == 'category']

    # 4. ביצוע One-Hot Encoding ידני
    df_proc = pd.get_dummies(df_proc, columns=cat_cols_proc)

    # 5. ביצוע Scaling על הפיצ'רים המספריים
    df_proc.loc[:, num_cols_proc] = telco_scaler.transform(df_proc[num_cols_proc])

    # 6. התאמת העמודות לסדר האימון
    # מוודאים שכל העמודות שהמודל מכיר קיימות, ואם לא, מוסיפים אותן עם ערכים של 0
    missing_cols = set(final_columns) - set(df_proc.columns)
    for c in missing_cols:
        df_proc[c] = 0
    df_proc = df_proc[final_columns]

    print(">> הנתונים החדשים עובדו בהצלחה והותאמו למודל.")

    # 7. ביצוע חיזוי
    prediction = telco_model.predict(df_proc)[0]
    probability = telco_model.predict_proba(df_proc)[:, 1][0]

    return prediction, probability


# --- דוגמה לשימוש בפונקציה ---
if __name__ == "__main__":
    # יצירת DataFrame של נתונים חדשים (לצורך הדגמה)
    new_telco_customer = pd.DataFrame({
        'customerID': ['0000-NEW-CUST'],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [29.85],
        'TotalCharges': [29.85]
    })

    pred, proba = predict_telco_churn(new_telco_customer)

    if pred is not None:
        print("\n--- תוצאות החיזוי ---")
        print(f"הלקוח צפוי {'' if pred == 0 else 'ל'}{'א ' if pred == 0 else ''}נטישה.")
        print(f"הסתברות לנטישה: {proba:.2f}")