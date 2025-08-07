from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_and_save_data(output_dir='data'):
    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    file_path = os.path.join(output_dir, 'california_housing.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

if __name__ == "__main__":
    load_and_save_data()
