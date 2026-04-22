"""
Download Kaggle Playground Series S6E4 - Predicting Irrigation Need
Run this script first to download the competition data.
You need a kaggle.json API key file in ~/.kaggle/
"""
import os
import sys
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')

def download_from_kaggle():
    """Download using Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Downloading competition data...")
        api.competition_download_files(
            'playground-series-s6e4',
            path=DATA_DIR,
            quiet=False
        )
        # Extract zip if needed
        zip_path = os.path.join(DATA_DIR, 'playground-series-s6e4.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            os.remove(zip_path)
            print("Data extracted successfully!")
        return True
    except Exception as e:
        print(f"Kaggle API failed: {e}")
        return False

def check_data_exists():
    """Check if train.csv and test.csv exist."""
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    return os.path.exists(train_path) and os.path.exists(test_path)

if __name__ == '__main__':
    if check_data_exists():
        print("Data already exists in Data/ folder!")
        sys.exit(0)
    
    print("train.csv and test.csv not found in Data/ folder.")
    print("\nOption 1: Download from Kaggle API")
    print("  - Place your kaggle.json in ~/.kaggle/")
    print("  - Run this script again")
    print("\nOption 2: Manual download")
    print("  - Go to: https://www.kaggle.com/competitions/playground-series-s6e4/data")
    print("  - Download train.csv and test.csv")
    print(f"  - Place them in: {os.path.abspath(DATA_DIR)}")
    
    print("\nAttempting Kaggle API download...")
    if not download_from_kaggle():
        print("Please download manually from the Kaggle competition page.")
