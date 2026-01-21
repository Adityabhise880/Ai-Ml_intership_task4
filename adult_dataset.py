import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. SETUP & LOAD ---
# Find the file automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'adult.csv')

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'income']

try:
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path, names=columns, na_values=' ?', skipinitialspace=True)
except FileNotFoundError:
    print("ERROR: File not found.")
    exit()

# --- 2. CRITICAL FIX: FORCE NUMBERS TO BE NUMBERS ---
# This prevents the 'KeyError' by making sure 'age' stays a number column
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that got messed up (became NaN)
df.dropna(inplace=True)

# --- 3. ENCODING ---
# Label Encode Target
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

# One-Hot Encode Categories (Drop 'education' as we have 'education-num')
df = df.drop('education', axis=1) 
df_encoded = pd.get_dummies(df, drop_first=True)

# --- 4. SCALING ---
# Now this will work because the columns are definitely there!
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# --- 5. SAVE ---
output_path = os.path.join(script_dir, 'adult_preprocessed.csv')
df_encoded.to_csv(output_path, index=False)

print(f"SUCCESS: Saved to {output_path}")