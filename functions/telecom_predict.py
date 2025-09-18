import joblib
import pandas as pd
import numpy as np

# --- הגדרת שמות הקבצים ---
MODEL_FILE_NAME = "../saved_models/telecom.pkl"
PREPROCESSOR_DATA_FILE = "../preprocessing/telecom/telecom_preprocessor_data.pkl"


def predict_telecom_churn(new_data: pd.DataFrame) -> tuple:
    """
    מבצע חיזוי נטישה על נתוני לקוחות Telecom חדשים באמצעות Pipeline מובנה.

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

    # 1. טעינת המודל וה-preprocessor מהקבצים
    try:
        telecom_model = joblib.load(MODEL_FILE_NAME)
        preprocessor_data = joblib.load(PREPROCESSOR_DATA_FILE)
        telecom_preprocessor = preprocessor_data["preprocessor"]
        print(">> מודל Telecom וקבצי Preprocessing נטענו בהצלחה.")
    except FileNotFoundError as e:
        print(f"שגיאה: קובץ לא נמצא - {e}. ודא שהקבצים נמצאים באותו נתיב.")
        return None, None

    # 2. ביצוע Preprocessing על הנתונים החדשים באמצעות ה-Pipeline
    print(">> מבצע Preprocessing על הנתונים החדשים באמצעות Pipeline...")
    # ה-pipeline מטפל באופן אוטומטי ב-scaling ו-one-hot encoding
    new_data_processed = telecom_preprocessor.transform(new_data)

    print(">> הנתונים החדשים עובדו בהצלחה.")

    # 3. ביצוע חיזוי
    prediction = telecom_model.predict(new_data_processed)[0]
    probability = telecom_model.predict_proba(new_data_processed)[:, 1][0]

    return prediction, probability


# --- דוגמה לשימוש בפונקציה ---
if __name__ == "__main__":
    # יצירת DataFrame של נתונים חדשים (לצורך הדגמה)
    new_telecom_customer = pd.DataFrame({
        'State': ['NY'],
        'Account length': [120],
        'Area code': [415],
        'International plan': ['No'],
        'Voice mail plan': ['Yes'],
        'Number vmail messages': [25],
        'Total day minutes': [180.0],
        'Total day calls': [100],
        'Total day charge': [30.6],
        'Total eve minutes': [200.0],
        'Total eve calls': [105],
        'Total eve charge': [17.0],
        'Total night minutes': [220.0],
        'Total night calls': [110],
        'Total night charge': [9.9],
        'Total intl minutes': [10.0],
        'Total intl calls': [5],
        'Total intl charge': [2.7],
        'Customer service calls': [2]
    })

    pred, proba = predict_telecom_churn(new_telecom_customer)

    if pred is not None:
        print("\n--- תוצאות החיזוי ---")
        print(f"הלקוח צפוי {'' if pred == 0 else 'ל'}{'א ' if pred == 0 else ''}נטישה.")
        print(f"הסתברות לנטישה: {proba:.2f}")