import pandas as pd
import numpy as np

def load_data(filepath):
    #handle missing values
    df = pd.read_csv(filepath, na_values='?')
    return df

def map_diagnosis_codes(df):
    #map ICD-9 codes in diag_1, 2, 3
    print('Mapping Diag codes')

    def map_single_code(icd_code):
        if pd.isna(icd_code):
            return 'Missing'
        
        if str(icd_code).startswith(('V', 'E')):
            return 'Other'
        
        try:
            code = float(icd_code)
        except ValueError:
            return 'Other'


        if 390 <= code <= 459 or code == 785:
            return 'Circulatory'
        elif 460 <= code <= 519 or code == 786:
            return 'Respiratory'
        elif 520 <= code <= 579 or code == 787:
            return 'Digestive'
        elif 250 <= code < 251:
            return 'Diabetes'
        elif 800 <= code <= 999:
            return 'Injury'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif 580 <= code <= 629 or code == 788:
            return 'Genitourinary'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
        
    for col in ['diag_1', 'diag_2', 'diag_3']:
            df[col] = df[col].apply(map_single_code)

    return df

def clean_categorical_features(df):
    print('Cleaning categorical features')

    #drop columns with too many missing values
    cols_to_drop = ['weight', 'payer_code', 'encounter_id', 'patient_nbr']
    df = df.drop(columns=cols_to_drop, errors = 'ignore')

    #clean medical_specialty
    df['medical_specialty'] = df['medical_specialty'].fillna('Missing')

    top_10 = df['medical_specialty'].value_counts().index[:10]
    df.loc[~df['medical_specialty'].isin(top_10), 'medical_specialty'] = 'Other'

    return df

#target variable in readmission
def encode_target(df):
    print('Encoding target variable')

    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    #drop orginal readmitted column to avoid leakage
    df = df.drop(columns=['readmitted'])

    return df

if __name__ == "__main__":
    #file path
    RAW_PATH = 'data/raw/diabetic_data.csv'

    df = load_data(RAW_PATH)
    df = map_diagnosis_codes(df)
    df = clean_categorical_features(df)
    df = encode_target(df)

    #check result
    print('\n--- Preprocessing Complete ---')
    print(f"Shape: {df.shape}")
    print(f"Target Distribution:\n{df['target'].value_counts(normalize=True)}")

    #save processed data
    df.to_csv('data/processed_data.csv', index=False)
    print("saved to data/processed_data.csv")