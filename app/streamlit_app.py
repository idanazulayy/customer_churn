import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from arabic_reshaper import arabic_reshaper
from bidi import get_display
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import sys
import os

from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from functions.telco_predict import fe_total_charges
from functions.telco_predict import predict_telco_churn
from functions.bank_predict import predict_bank_churn
from functions.telecom_predict import predict_telecom_churn

# ------------------------------------------------
# הגדרות ראשוניות של העמוד
# ------------------------------------------------
st.set_page_config(
    page_title="מערכת לחיזוי נטישת לקוחות",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- הסתרת Footer של Streamlit ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
rtl_style = """
<style>
/* הפוך את כל הדף ל-RTL */

/* אפשר גם לכיווני כפתורים וטקסטים */
.stButton>button {
    direction: rtl;
}
.stAppViewContainer {
    direction: rtl;
}

/* לדוגמא, תיבות סלקט ו-input */
input, select, textarea {
    direction: rtl;
    text-align: right;
}

/* כותרות */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
}
</style>
"""
st.markdown(rtl_style, unsafe_allow_html=True)
st.session_state.clear()


def train_model(model_choice: str, train_df: pd.DataFrame):
    """
    מאמן מחדש את המודל על דאטה חדש.

    :param model_choice: שם המודל ("Bank" או "Telecom")
    :param train_df: DataFrame עם כל העמודות הנדרשות כולל 'target'
    :return: model חדש ו-ROC-AUC
    """
    if model_choice == "Bank":
        # טעינת preprocessor קיים
        preprocessor = joblib.load("../preprocessing/bank/bank_preprocessor_data.pkl")['preprocessor']

        X_train = train_df.drop(columns=['churn'])
        y_train = train_df['churn']

        # עיבוד
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = X_train_proc  # זמנית, אפשר להוסיף train/test split
        y_test = y_train

        # אימון מודל XGBoost
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],
            use_label_encoder=False,
            n_estimators=1750,
            reg_lambda=1.0,
            reg_alpha=0.7,
            max_depth=7,
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.6,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            random_state=42
        )
        model.fit(X_train_proc, y_train, eval_set=[(X_train_proc, y_train)], verbose=False)

        # חישוב ROC-AUC
        auc = roc_auc_score(y_test, model.predict_proba(X_test_proc)[:, 1])

        # שמירת preprocessor ומודל
        joblib.dump(model, "../new_models/bank_model.pkl")

        return model, auc

    elif model_choice == "Telecom":
        preprocessor = joblib.load("../preprocessing/telecom/telecom_preprocessor_data.pkl")['preprocessor']

        X_train = train_df.drop(columns=['Churn'])
        y_train = train_df['Churn']

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = X_train_proc
        y_test = y_train

        model = XGBClassifier(
            n_estimators=450,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
        model.fit(X_train_proc, y_train, eval_set=[(X_train_proc, y_train)], verbose=False)

        auc = roc_auc_score(y_test, model.predict_proba(X_test_proc)[:, 1])

        joblib.dump(model, "../new_models/telecom_model.pkl")

        return model, auc

    else:
        raise ValueError(f"אימון מחדש לא נתמך עבור מודל {model_choice}")
def fix_hebrew_text(text):
    """
    מבצעת שינוי צורה ותיקון כיוון של טקסט בעברית
    כדי שיוצג כראוי בפלוטים של matplotlib.
    """
    # יצירת מופע חדש של reshaper עבור כל קריאה
    reshaper = arabic_reshaper.ArabicReshaper({
        'delete_harakat': True,
        'support_ligatures': True
    })
    reshaped_text = reshaper.reshape(text)
    return get_display(reshaped_text)

# ------------------------------------------------
# טעינת מודלים ונתונים
# ------------------------------------------------
new_models_trained = {}

@st.cache_resource
def load_all_models():
    """טוען את כל המודלים והנתונים הנלווים כדי למנוע טעינה חוזרת."""
    models = {
        "Telco": joblib.load("../saved_models/telco.pkl"),
        "Bank": joblib.load("../saved_models/bank2.pkl"),
        "Telecom": joblib.load("../saved_models/telecom.pkl")
    }
    # טעינת קבצי ה-X_test ו-y_test שהועלו
    test_data = {
        "Telco": (pd.read_csv("../data/telco_X_test.csv"), pd.read_csv("../data/telco_y_test.csv")),
        "Bank": (pd.read_csv("../data/bank_X_test.csv"), pd.read_csv("../data/bank_y_test.csv")),
        "Telecom": (pd.read_csv("../data/telecom_X_test.csv"), pd.read_csv("../data/telecom_y_test.csv"))
    }

    # --- ביצוע preprocessing לכל נתוני המבחן ---
    # Telco
    telco_X_test_raw, telco_y_test = test_data["Telco"]
    from functions.telco_predict import fe_total_charges

    telco_scaler = joblib.load("../preprocessing/telco/telco_scaler.pkl")
    with open("../preprocessing/telco/telco_numerical_cols.json", 'r') as f:
        numerical_cols_to_scale = json.load(f)
    with open("../preprocessing/telco/telco_columns.json", 'r') as f:
        final_columns = json.load(f)

    telco_X_test_proc = telco_X_test_raw.copy()
    telco_X_test_proc = fe_total_charges(telco_X_test_proc)

    # Feature Engineering כפי שמוגדר בקובץ telco_predict.py
    telco_X_test_proc['AverageMonthlyCharge'] = telco_X_test_proc['TotalCharges'] / telco_X_test_proc['tenure']
    telco_X_test_proc['AverageMonthlyCharge'] = telco_X_test_proc['AverageMonthlyCharge'].replace([np.inf, -np.inf], 0)
    telco_X_test_proc['AverageMonthlyCharge'] = telco_X_test_proc['AverageMonthlyCharge'].fillna(0)

    telco_X_test_proc['tenure_group'] = pd.cut(telco_X_test_proc['tenure'],
                                               bins=[0, 12, 24, 48, np.inf],
                                               labels=['0-12', '13-24', '25-48', '49+'],
                                               right=False)
    telco_X_test_proc['HasInternetService'] = telco_X_test_proc['InternetService'].apply(
        lambda x: 1 if x in ['DSL', 'Fiber optic'] else 0
    )
    extra_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies']
    telco_X_test_proc['Num_Extra_Services'] = telco_X_test_proc[extra_services].apply(
        lambda row: sum(1 for service in row if service == 'Yes'), axis=1
    )
    telco_X_test_proc['SeniorCitizen_x_Contract'] = telco_X_test_proc.apply(
        lambda row: f"{row['SeniorCitizen']}_{row['Contract']}", axis=1
    )

    cat_cols_proc = [col for col in telco_X_test_proc.columns if
                     col not in numerical_cols_to_scale and telco_X_test_proc.dtypes[col] == 'object' or
                     telco_X_test_proc.dtypes[col].name == 'category']
    telco_X_test_proc = pd.get_dummies(telco_X_test_proc, columns=cat_cols_proc)

    # לוודא שהעמודות המספריות הן מסוג float לפני הטרנספורמציה
    telco_X_test_proc.loc[:, numerical_cols_to_scale] = telco_X_test_proc.loc[:, numerical_cols_to_scale].astype(float)
    telco_X_test_proc.loc[:, numerical_cols_to_scale] = telco_scaler.transform(
        telco_X_test_proc[numerical_cols_to_scale])

    missing_cols = set(final_columns) - set(telco_X_test_proc.columns)
    for c in missing_cols:
        telco_X_test_proc[c] = 0
    telco_X_test_proc = telco_X_test_proc[final_columns]

    telco_y_test = telco_y_test.iloc[:, 0]
    test_data["Telco"] = (telco_X_test_proc, telco_y_test)

    # Bank
    bank_X_test_raw, bank_y_test = test_data["Bank"]
    bank_preprocessor_data = joblib.load("../preprocessing/bank/bank_preprocessor_data.pkl")
    bank_preprocessor = bank_preprocessor_data['preprocessor']
    bank_X_test_proc = bank_preprocessor.transform(bank_X_test_raw)
    bank_y_test = bank_y_test.iloc[:, 0]
    test_data["Bank"] = (bank_X_test_proc, bank_y_test)

    # Telecom
    telecom_X_test_raw, telecom_y_test = test_data["Telecom"]
    telecom_preprocessor_data = joblib.load("../preprocessing/telecom/telecom_preprocessor_data.pkl")
    telecom_preprocessor = telecom_preprocessor_data["preprocessor"]
    telecom_X_test_proc = telecom_preprocessor.transform(telecom_X_test_raw)
    telecom_y_test = telecom_y_test.iloc[:, 0]
    test_data["Telecom"] = (telecom_X_test_proc, telecom_y_test)

    return models, test_data


@st.cache_resource
def load_eda_data():
    """טוען את ה-DataFrame המלא עבור ה-EDA מהקבצים הקיימים."""
    try:
        telco_X_test = pd.read_csv("../data/telco_X_test.csv")
        telco_y_test = pd.read_csv("../data/telco_y_test.csv").iloc[:, 0]
        telco_df_eda = pd.concat([telco_X_test, telco_y_test], axis=1)

        bank_X_test = pd.read_csv("../data/bank_X_test.csv")
        bank_y_test = pd.read_csv("../data/bank_y_test.csv").iloc[:, 0]
        bank_df_eda = pd.concat([bank_X_test, bank_y_test], axis=1)
        bank_df_eda.rename(columns={'Exited': 'churn'}, inplace=True)

        telecom_X_test = pd.read_csv("../data/telecom_X_test.csv")
        telecom_y_test = pd.read_csv("../data/telecom_y_test.csv").iloc[:, 0]
        telecom_df_eda = pd.concat([telecom_X_test, telecom_y_test], axis=1)
        telecom_df_eda.rename(columns={'Churn': 'Churn'}, inplace=True)

        return {
            "Telco": telco_df_eda,
            "Bank": bank_df_eda,
            "Telecom": telecom_df_eda
        }
    except FileNotFoundError:
        st.error("שגיאה: קבצי הנתונים לא נמצאו בתיקיית '../data'. ודא שהם קיימים.")
        return None


models, test_data = load_all_models()
eda_data = load_eda_data()

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("ניווט בפרויקט")
page = st.sidebar.radio(
    "בחר עמוד:",
    ["🏠 ראשי", "📊 ניתוח נתונים (EDA)", "🔮 חיזוי עבור לקוח בודד", "📂 חיזוי על קובץ", "📈 הערכת ביצועי מודלים"]
)

# ------------------------------------------------
# עמוד ראשי - Dashboard
# ------------------------------------------------
if page == "🏠 ראשי":
    st.title("🏠 מערכת לחיזוי נטישת לקוחות")
    st.markdown(
        "ברוכים הבאים לפרויקט הגמר שלי! מערכת זו מאגדת שלושה מודלים של למידת מכונה לחיזוי נטישת לקוחות בתחומים שונים: **תקשורת (Telco), בנקאות (Bank) ותקשורת נוספת (Telecom).**")
    st.markdown("ניתן להשתמש בסרגל הניווט בצד כדי לעבור בין חלקי המערכת השונים.")

    st.header("סקירת המודלים וביצועיהם")

    # חישוב מטריקות מנתוני המבחן האמיתיים
    metrics_data = []

    for model_name, (X_test, y_test) in test_data.items():
        model = models[model_name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics_data.append({
            "Model": model_name,
            "Accuracy": f"{accuracy:.2%}",
            "ROC-AUC": f"{roc_auc:.2f}"
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index("Model"))

    # גרף השוואת ביצועים
    st.subheader("השוואת ROC-AUC בין המודלים")
    fig = px.bar(metrics_df, x="Model", y="ROC-AUC", color="Model",
                 title="ROC-AUC Score by Model", text='ROC-AUC')
    fig.update_layout(yaxis_title="ROC-AUC Score", xaxis_title="Model")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# עמוד ניתוח נתונים - EDA
# ------------------------------------------------
elif page == "📊 ניתוח נתונים (EDA)":
    st.title("📊 ניתוח נתונים גישוש (EDA)")
    st.info("בעמוד זה מוצג ניתוח הנתונים המרכזי שבוצע על נתוני המבחן המייצגים.")

    if eda_data is None:
        st.stop()

    model_choice_eda = st.selectbox("בחר דאטהסט להצגת ניתוח נתונים:", list(eda_data.keys()))
    df_eda = eda_data[model_choice_eda]

    if model_choice_eda == "Telco":
        st.header("ניתוח נתוני לקוחות Telco")
        df_eda['TotalCharges'] = pd.to_numeric(df_eda['TotalCharges'], errors='coerce')
        df_eda['TotalCharges'].fillna(0, inplace=True)
        # התפלגות נטישה
        churn_count = df_eda['Churn'].value_counts()
        fig_churn, ax_churn = plt.subplots()
        sns.countplot(x='Churn', data=df_eda, ax=ax_churn)
        ax_churn.set_title(fix_hebrew_text("התפלגות נטישה"))
        ax_churn.set_xticklabels([fix_hebrew_text('לא נוטש'), fix_hebrew_text('נוטש')])
        st.pyplot(fig_churn)
        st.write(f"התפלגות הנטישה: {churn_count[1]} נוטשים, {churn_count[0]} לא נוטשים.")

        # התפלגות חודשים בשירות מול נטישה
        st.subheader("קשר בין ותק (tenure) לנטישה")
        fig_tenure, ax_tenure = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_eda, x='tenure', hue='Churn', kde=True, ax=ax_tenure, bins=30)
        ax_tenure.set_title(fix_hebrew_text("התפלגות הותק לפי נטישה"))
        st.pyplot(fig_tenure)
        st.info("ניתן לראות שרוב הלקוחות שנוטשים הם לקוחות חדשים יחסית (בעלי ותק קצר).")

        # התפלגות עלויות חודשיות מול נטישה
        st.subheader("קשר בין חיובים חודשיים (MonthlyCharges) לנטישה")
        fig_charges, ax_charges = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_eda, x='MonthlyCharges', hue='Churn', kde=True, ax=ax_charges, bins=30)
        ax_charges.set_title(fix_hebrew_text("התפלגות החיובים החודשיים לפי נטישה"))
        st.pyplot(fig_charges)
        st.info("לקוחות עם חיובים חודשיים גבוהים נוטים לנשור יותר.")

        # השפעת שירותים שונים על נטישה
        st.subheader("השפעת שירותים וחוזים על נטישה")
        cat_features = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
        for col in cat_features:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=col, hue='Churn', data=df_eda, ax=ax)
            ax.set_title(fix_hebrew_text(f"קשר בין {col} לנטישה"))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        # מטריצת קורלציה
        num_cols = df_eda.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.subheader("מטריצת קורלציה")
        corr = df_eda[num_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title(fix_hebrew_text("מטריצת קורלציה בין משתנים"))
        st.pyplot(fig_corr)
        st.info("מטריצת קורלציה מציגה את הקשרים הלינאריים בין המשתנים השונים.")

    elif model_choice_eda == "Bank":
        st.header("ניתוח נתוני לקוחות בנק")
        # התפלגות נטישה
        churn_count = df_eda['churn'].value_counts()
        fig_churn, ax_churn = plt.subplots()
        sns.countplot(x='churn', data=df_eda, ax=ax_churn)
        ax_churn.set_title(fix_hebrew_text("התפלגות נטישה"))
        ax_churn.set_xticklabels([fix_hebrew_text('לא נוטש'), fix_hebrew_text('נוטש')])
        st.pyplot(fig_churn)
        st.write(f"התפלגות הנטישה: {churn_count[1]} נוטשים, {churn_count[0]} לא נוטשים.")

        # התפלגות עמודות מספריות
        st.subheader("התפלגות של משתנים מספריים")
        num_cols = df_eda.select_dtypes(include=['int64', 'float64']).columns.tolist()
        num_cols.remove('churn')
        for col in num_cols:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df_eda[col], kde=True, ax=ax_hist)
            ax_hist.set_title(fix_hebrew_text(f"התפלגות {col}"))
            st.pyplot(fig_hist)

        # מטריצת קורלציה
        st.subheader("מטריצת קורלציה")
        corr = df_eda[num_cols + ['churn']].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title(fix_hebrew_text("מטריצת קורלציה בין משתנים"))
        st.pyplot(fig_corr)
        st.info("מטריצת קורלציה מציגה את הקשרים הלינאריים בין המשתנים השונים.")

        # קשר בין מדינה לנטישה
        st.subheader("קשר בין מדינה לנטישה")
        fig_country = px.bar(
            df_eda.groupby('country')['churn'].mean().reset_index(),
            x='country', y='churn', title='שיעור נטישה לפי מדינה',
            labels={'churn': 'שיעור נטישה', 'country': 'מדינה'}
        )
        st.plotly_chart(fig_country)

    elif model_choice_eda == "Telecom":
        st.header("ניתוח נתוני לקוחות תקשורת (Telecom)")
        # התפלגות נטישה
        churn_count = df_eda['Churn'].value_counts()
        fig_churn, ax_churn = plt.subplots()
        sns.countplot(x='Churn', data=df_eda, ax=ax_churn)
        ax_churn.set_title(fix_hebrew_text("התפלגות נטישה"))
        ax_churn.set_xticklabels([fix_hebrew_text('לא נוטש'), fix_hebrew_text('נוטש')])
        st.pyplot(fig_churn)
        st.write(f"התפלגות הנטישה: {churn_count[1]} נוטשים, {churn_count[0]} לא נוטשים.")

        # התפלגות חיובים לפי יום/ערב/לילה
        st.subheader("התפלגות חיובים לפי זמן שיחה")
        fig_dist = px.box(df_eda, y=['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'],
                          title="התפלגות החיובים לפי שעות היום")
        st.plotly_chart(fig_dist)
        st.info("ניתן לראות שהחיובים ביום הם הגבוהים ביותר.")

        # השפעת שיחות לשירות לקוחות על נטישה
        st.subheader("השפעת שיחות שירות לקוחות על נטישה")
        fig_cs = px.bar(
            df_eda.groupby('Customer service calls')['Churn'].mean().reset_index(),
            x='Customer service calls', y='Churn',
            title='שיעור נטישה לפי מספר שיחות לשירות לקוחות',
            labels={'Churn': 'שיעור נטישה'}
        )
        st.plotly_chart(fig_cs)
        st.info("ככל שלקוח מבצע יותר שיחות לשירות לקוחות, כך שיעור הנטישה עולה משמעותית.")
        # מטריצת קורלציה
        num_cols = df_eda.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.subheader("מטריצת קורלציה")
        corr = df_eda[num_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title(fix_hebrew_text("מטריצת קורלציה בין משתנים"))
        st.pyplot(fig_corr)
        st.info("מטריצת קורלציה מציגה את הקשרים הלינאריים בין המשתנים השונים.")

# ------------------------------------------------
# עמוד חיזוי ללקוח בודד
# ------------------------------------------------
elif page == "🔮 חיזוי עבור לקוח בודד":
    st.title("🔮 חיזוי והסבר עבור לקוח בודד")

    model_choice = st.selectbox("בחר מודל לחיזוי:", list(models.keys()))

    st.header(f"הזן את פרטי הלקוח עבור מודל {model_choice}")

    # --- טופס קלט דינאמי על פי המודל הנבחר ---
    if model_choice == "Telco":
        # טופס עבור Telco
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input(" ותק (חודשים)", min_value=0, max_value=100, value=1)
            MonthlyCharges = st.number_input("חיוב חודשי", min_value=0.0, value=30.0)
            Contract = st.selectbox("סוג חוזה", ['Month-to-month', 'One year', 'Two year'])
        with col2:
            InternetService = st.selectbox("שירות אינטרנט", ['DSL', 'Fiber optic', 'No'])
            OnlineSecurity = st.selectbox("אבטחת אונליין", ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox("תמיכה טכנית", ['Yes', 'No', 'No internet service'])
        with col3:
            gender = st.selectbox("מגדר", ['Male', 'Female'])
            PaperlessBilling = st.selectbox("חיוב ללא נייר", ['Yes', 'No'])
            PaymentMethod = st.selectbox("שיטת תשלום", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                        'Credit card (automatic)'])

        # יצירת DataFrame לדוגמה - יש להשלים את כל הפיצ'רים הנדרשים!
        new_customer_df = pd.DataFrame({
            'customerID': ['0000-NEW'], 'gender': [gender], 'SeniorCitizen': [0], 'Partner': ['No'],
            'Dependents': ['No'], 'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'],
            'InternetService': [InternetService], 'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': ['No'],
            'DeviceProtection': ['No'], 'TechSupport': ['No'], 'StreamingTV': ['No'],
            'StreamingMovies': ['No'], 'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [MonthlyCharges * tenure]
        })

        predict_func = predict_telco_churn

    elif model_choice == "Bank":
        # טופס עבור Bank
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("ניקוד אשראי", 300, 850, 600)
            age = st.slider("גיל", 18, 100, 40)
            tenure = st.slider("ותק בחשבון (שנים)", 0, 10, 3)
        with col2:
            balance = st.number_input("יתרה בחשבון", value=60000.0)
            products_number = st.selectbox("מספר מוצרים", [1, 2, 3, 4])
            active_member = st.selectbox("האם לקוח פעיל?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        with col3:
            country = st.selectbox("מדינה", ['Spain', 'France', 'Germany'])
            gender = st.selectbox("מגדר", ['Male', 'Female'])
            credit_card = st.selectbox("האם יש כרטיס אשראי?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        new_customer_df = pd.DataFrame({
            'credit_score': [credit_score], 'country': [country], 'gender': [gender], 'age': [age],
            'tenure': [tenure], 'balance': [balance], 'products_number': [products_number],
            'credit_card': [credit_card], 'active_member': [active_member], 'estimated_salary': [50000]
        })

        predict_func = predict_bank_churn

    elif model_choice == "Telecom":
        # טופס עבור Telecom
        col1, col2, col3 = st.columns(3)
        with col1:
            Account_length = st.number_input("ותק חשבון", min_value=0, value=120)
            International_plan = st.selectbox("תוכנית בינלאומית", ['No', 'Yes'])
            Voice_mail_plan = st.selectbox("תוכנית תא קולי", ['No', 'Yes'])
        with col2:
            Total_day_minutes = st.number_input("דקות שיחה (יום)", value=180.0)
            Total_eve_minutes = st.number_input("דקות שיחה (ערב)", value=200.0)
            Total_night_minutes = st.number_input("דקות שיחה (לילה)", value=220.0)
        with col3:
            Customer_service_calls = st.number_input("שיחות לשירות לקוחות", min_value=0, value=2)
            Total_intl_calls = st.number_input("שיחות בינלאומיות", min_value=0, value=5)

        # יצירת DataFrame מלא. יש לוודא שכל העמודות שהמודל מצפה להן קיימות.
        new_customer_df = pd.DataFrame({
            'State': ['NY'], 'Account length': [Account_length], 'Area code': [415],
            'International plan': [International_plan], 'Voice mail plan': [Voice_mail_plan],
            'Number vmail messages': [25 if Voice_mail_plan == 'Yes' else 0],
            'Total day minutes': [Total_day_minutes], 'Total day calls': [100],
            'Total day charge': [Total_day_minutes * 0.17],
            'Total eve minutes': [Total_eve_minutes], 'Total eve calls': [105],
            'Total eve charge': [Total_eve_minutes * 0.085],
            'Total night minutes': [Total_night_minutes], 'Total night calls': [110],
            'Total night charge': [Total_night_minutes * 0.045],
            'Total intl minutes': [10.0], 'Total intl calls': [Total_intl_calls], 'Total intl charge': [2.7],
            'Customer service calls': [Customer_service_calls]
        })

        predict_func = predict_telecom_churn

    if st.button("🔮 בצע חיזוי", key=f"predict_{model_choice}"):
        prediction, probability = predict_func(new_customer_df)

        churn_status = "נוטש" if prediction == 1 else "לא נוטש"
        color = "red" if prediction == 1 else "green"

        st.subheader("תוצאת החיזוי:")
        st.markdown(f"הלקוח צפוי **<span style='color:{color};'>{churn_status}</span>**.", unsafe_allow_html=True)

        st.metric(label="הסתברות לנטישה", value=f"{probability:.2%}")

        # --- XAI עם SHAP ---
        st.subheader("הסבר לחיזוי (Explainable AI - SHAP)")
        with st.spinner("מחשב ערכי SHAP..."):
            model = models[model_choice]

            # טעינת ה-preprocessor המתאים
            if model_choice == "Bank":
                preprocessor = joblib.load("../preprocessing/bank/bank_preprocessor_data.pkl")['preprocessor']
                processed_data = preprocessor.transform(new_customer_df)
                feature_names = preprocessor.get_feature_names_out()
            elif model_choice == "Telecom":
                preprocessor = joblib.load("../preprocessing/telecom/telecom_preprocessor_data.pkl")['preprocessor']
                processed_data = preprocessor.transform(new_customer_df)
                feature_names = preprocessor.get_feature_names_out()
            else:  # Telco (שם אין preprocessor מאוחד)
                st.info(
                    "הסברי SHAP עבור מודל Telco דורשים טיפול מיוחד בגלל ה-preprocessing הידני ואינם זמינים כרגע בדמו זה.")
                st.stop()

            # יצירת Explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_data)

            # בוחרים את השורה הראשונה (לקוח יחיד)
            if isinstance(shap_values, list):  # בינארי
                shap_row = shap_values[1][0]
                expected_value = explainer.expected_value[1]
            else:  # רגרסיה
                shap_row = shap_values[0]
                expected_value = explainer.expected_value

            # --- גרף waterfall (הכי ברור לפרויקט) ---
            st.write("תרשים זה מציג אילו פיצ'רים תרמו לנטישה (אדום) ואילו מנעו נטישה (כחול).")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots._waterfall.waterfall_legacy(expected_value,
                                                   shap_row,
                                                   feature_names=feature_names)
            st.pyplot(fig, bbox_inches="tight")

            # --- גרף בר (top features) ---
            st.write("פיצ'רים הכי משפיעים על החלטת המודל:")

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(
                shap_row.reshape(1, -1),
                features=processed_data,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)



# ------------------------------------------------
# עמוד העלאת קובץ
# ------------------------------------------------
elif page == "📂 חיזוי על קובץ":
    st.title(fix_hebrew_text(fix_hebrew_text("📂 העלאת קובץ CSV לחיזוי")))

    uploaded_file = st.file_uploader("בחר קובץ CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # שינוי כאן: שימוש ב-st.dataframe() במקום st.write(df.head())
        st.write(fix_hebrew_text(fix_hebrew_text("תצוגה מקדימה של הנתונים:")))
        st.dataframe(df.head())  # אפשר להשאיר את head() כדי להציג רק את ההתחלה

        model_choice_upload = st.selectbox("בחר מודל להרצת החיזויים:", list(models.keys()))

        if st.button("הרץ חיזוי על הקובץ"):
            with st.spinner(fix_hebrew_text("מעבד את הקובץ ומבצע חיזוי...")):
                if model_choice_upload == "Telco":
                    predict_func_batch = predict_telco_churn
                elif model_choice_upload == "Bank":
                    predict_func_batch = predict_bank_churn
                else:
                    predict_func_batch = predict_telecom_churn

                # בדיקה אם יש נתונים
                if not df.empty:
                    # שימוש ב-iterrows עבור ביצועים טובים יותר על DataFrames קטנים-בינוניים
                    results = [predict_func_batch(pd.DataFrame([row])) for index, row in df.iterrows()]
                    predictions = [res[0] for res in results]
                    probabilities = [res[1] for res in results]

                    df["Prediction"] = predictions
                    df["Churn Probability"] = probabilities

                    st.success(fix_hebrew_text(fix_hebrew_text("החיזוי הושלם בהצלחה!")))
                    st.write("תוצאות:")

                    # שינוי נוסף: שימוש ב-st.dataframe() להצגת כל ה-DataFrame עם גלילה
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        fix_hebrew_text(fix_hebrew_text("הורד תוצאות")),
                        csv,
                        f"{model_choice_upload}_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning(fix_hebrew_text("הקובץ שהועלה ריק."))
# ------------------------------------------------
# עמוד הערכת מודלים
# ------------------------------------------------
elif page == "📈 הערכת ביצועי מודלים":
    st.title("📈 הערכת ביצועי מודלים")

    # בחירת מודל
    model_choice_eval = st.selectbox("בחר מודל להערכה:", list(models.keys()))
    model = models[model_choice_eval]
    X_test, y_test = test_data[model_choice_eval]

    # --- ביצועים נוכחיים ---
    st.header(f"ביצועי מודל {model_choice_eval} על נתוני מבחן קיימים")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    with col2:
        st.subheader("ROC-AUC Score")
        auc_score = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC-AUC", f"{auc_score:.4f}")
        st.subheader("Accuracy Score")
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Not Churn', 'Churn'],
        y=['Not Churn', 'Churn'],
        colorscale='Blues'
    )
    fig.update_layout(title_text='<i><b>Confusion Matrix</b></i>',
                      xaxis_title="Predicted value",
                      yaxis_title="Actual value")
    st.plotly_chart(fig, use_container_width=True)

    # --- Upload דאטה חדש לאימון מחדש ---
    st.subheader("🔄 אימון מחדש עם דאטה חדש")
    uploaded_file = st.file_uploader("בחר קובץ CSV עם העמודות הנכונות (כולל 'target')", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("דוגמאות מהדאטה שהועלה:")
        st.dataframe(new_data.head())

        if st.button(f"אמן מחדש את {model_choice_eval} עם הדאטה הזה"):
            with st.spinner("מאמן מחדש את המודל..."):
                new_model, new_auc = train_model(model_choice_eval, new_data)

            st.write(f"**ROC-AUC חדש:** {new_auc:.4f}")

            if new_auc > auc_score:
                st.success("✅ המודל החדש טוב יותר מהקיים!")

                # כפתור לאישור החלפה
                if st.button("החלף את המודל הקיים במודל החדש"):
                    new_models_trained[model_choice_eval] = new_model
                    st.success("✅ המודל הקיים עודכן בהצלחה.")
            else:
                st.warning("⚠ המודל החדש לא טוב יותר. המודל הקיים נשאר כפי שהוא.")