import pandas as pd

df_bank = pd.read_csv('bank_data.csv')
df_telco = pd.read_csv('telco_data.csv')
df_telecom = pd.read_csv('telecom_data.csv')

df_bank.drop(columns='customer_id',inplace=True)
df_telco.drop(columns='customerID',inplace=True)
# df_telecom.drop(columns='customerID',inplace=True)



#     ├── telecom_X_test.csv
#     └── telecom_y_test.csv
# bank_X_test = df_bank.drop(columns='churn')
# bank_y_test = df_bank['churn']
# bank_X_test.to_csv('bank_X_test.csv',index=False)
# bank_y_test.to_csv('bank_y_test.csv',index=False)

# telco_X_test = df_telco.drop(columns='Churn')
# telco_y_test = df_telco['Churn'].map({'No': 0, 'Yes': 1})
# telco_X_test.to_csv('telco_X_test.csv',index=False)
# telco_y_test.to_csv('telco_y_test.csv',index=False)

telecom_X_test = df_telecom.drop(columns='Churn')
telecom_y_test = df_telecom['Churn'].astype(int)  # False -> 0, True -> 1
telecom_X_test.to_csv('telecom_X_test.csv',index=False)
telecom_y_test.to_csv('telecom_y_test.csv',index=False)