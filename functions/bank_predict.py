import joblib
import pandas as pd
import os

# --- שם הקובץ של המודל וה-preprocessor ---
MODEL_FILE_NAME = "../saved_models/bank.pkl"
PREPROCESSOR_FILE_NAME = "../preprocessing/bank/bank_preprocessor_data.pkl"


def predict_bank_churn(new_data: pd.DataFrame) -> tuple:
    """
    מבצע חיזוי נטישה על נתוני לקוחות בנק חדשים.

    הפונקציה טוענת את המודל המאומן ואת ה-preprocessor, מעבדת את הנתונים החדשים
    ומחזירה את החיזוי הסופי וההסתברות.

    Parameters:
    ----------
    new_data : pd.DataFrame
        DataFrame המכיל את נתוני הלקוחות החדשים, עם אותן עמודות כמו נתוני האימון המקוריים.

    Returns:
    -------
    tuple (prediction, probability)
        prediction: 0 (לא נוטש) או 1 (נוטש).
        probability: ההסתברות לנטישה (ערך בין 0 ל-1).
    """
    # 1. טעינת המודל וה-preprocessor מהקבצים
    try:
        bank_model = joblib.load(MODEL_FILE_NAME)
        preprocessor_data = joblib.load(PREPROCESSOR_FILE_NAME)
        bank_preprocessor = preprocessor_data['preprocessor']
        print(">> מודל ומעבד מקדים של הבנק נטענו בהצלחה.")
    except FileNotFoundError as e:
        print(f"שגיאה: קובץ לא נמצא - {e}. ודא שהקבצים נמצאים באותו נתיב.")
        return None, None

    # 2. עיבוד מקדים של הנתונים החדשים
    try:
        # לוודא שהעמודות המספריות והקטגוריאליות קיימות ב-DataFrame החדש
        # ולבצע טרנספורמציה באמצעות ה-preprocessor
        processed_data = bank_preprocessor.transform(new_data)
        print(">> הנתונים החדשים עובדו בהצלחה.")
    except Exception as e:
        print(f"שגיאה בעיבוד הנתונים: {e}. ודא שמבנה הנתונים תואם את מבנה האימון.")
        return None, None

    # 3. ביצוע חיזוי
    prediction = bank_model.predict(processed_data)[0]
    probability = bank_model.predict_proba(processed_data)[:, 1][0]

    return prediction, probability


# --- דוגמה לשימוש בפונקציה ---
if __name__ == "__main__":
    # יצירת DataFrame של נתונים חדשים (לצורך הדגמה)
    # הערה: יש לוודא ששמות העמודות וסוגי הנתונים זהים לאלו ששימשו לאימון.
    new_bank_customer = pd.DataFrame({
        'credit_score': [600],
        'country': ['Spain'],
        'gender': ['Male'],
        'age': [40],
        'tenure': [3],
        'balance': [60000],
        'products_number': [2],
        'credit_card': [1],
        'active_member': [1],
        'estimated_salary': [50000]
    })

    pred, proba = predict_bank_churn(new_bank_customer)

    if pred is not None:
        print("\n--- תוצאות החיזוי ---")
        print(f"הלקוח צפוי {'' if pred == 0 else 'ל'}{'א ' if pred == 0 else ''}נטישה.")
        print(f"הסתברות לנטישה: {proba:.2f}")