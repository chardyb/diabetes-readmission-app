import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def explain_model_features():
    print("--- Extracting Feature Importance ---")

    #load data
    df = pd.read_csv("data/processed_data.csv")
    clf = joblib.load('data/diabetes_model.pkl')

    #extract pipeline components
    preprocessor = clf.named_steps['preprocessor']
    model = clf.named_steps['model']

    #original column names
    X = df.drop(columns=['target'])
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    #encoded names from onehotencoder
    ohe_feature_names =  preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

    feature_names = numerical_cols + list(ohe_feature_names)

    #get importance scores
    importances = model.feature_importances_

    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    #sort by importance 
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)

    #plot
    plt.figure(figsize=(10,8))

    plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()

    plt.title('Top 20 Factors Predicting Reasmission (SGBoost Feature Importance)')
    plt.xlabel('Importance Score')

    #save results
    save_path = 'results/feature_importance.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")

if __name__ == "__main__":
    explain_model_features()