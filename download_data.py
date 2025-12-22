import pandas as pd
import requests
import zipfile
import os
import io


def download_and_extract_data():
    # URL for the Diabetes 130-US hospitals dataset
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    
    print(f"Downloading data from {url}...")
    
    # Get the request
    response = requests.get(url)
    
    if response.status_code == 200:
        # Create a ZipFile object from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print("Extracting files...")
            z.extractall("data/raw") # Extract to a data/raw folder
            print("Extraction complete.")
            
            # List files to confirm
            print("\nFiles in 'data/raw':")
            for file in z.namelist():
                print(f" - {file}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def load_initial_data():
    # The actual data is usually in 'dataset_diabetes/diabetic_data.csv' 
    # (based on the zip structure)
    file_path = "data/raw/diabetic_data.csv"
    
    if os.path.exists(file_path):
        print(f"\nLoading {file_path} into Pandas...")
        # Note: The dataset uses '?' for missing values
        df = pd.read_csv(file_path, na_values='?')
        
        print("\n--- Data Loaded Successfully ---")
        print(f"Shape: {df.shape} (Rows, Columns)")
        print("\nFirst 5 Rows:")
        print(df.head())
        print("\nColumns with Missing Values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
        return df
    else:
        print(f"Error: Could not find file at {file_path}")
        return None

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    download_and_extract_data()
    df = load_initial_data()