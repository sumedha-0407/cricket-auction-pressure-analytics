from pathlib import Path
import pandas as pd

# Define data path
DATA_DIR = Path("data/raw")

def audit_csv(file_path, primary_key=None):
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"\n==============================")
    print(f"FILE: {file_path.name}")
    print(f"==============================")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Check missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nMissing Values:")
        print(missing)
    else:
        print("\nMissing Values: None")
        
    # Check primary key uniqueness if provided
    if primary_key and primary_key in df.columns:
        duplicates = df[primary_key].duplicated().sum()
        print(f"Primary Key ({primary_key}) Duplicates: {duplicates}")
    elif primary_key:
        print(f" Primary key '{primary_key}' not found in columns!")

if __name__ == "__main__":
    audit_csv(DATA_DIR / "competitions.csv", primary_key="competition_id")
    audit_csv(DATA_DIR / "matches.csv", primary_key="match_id")
    audit_csv(DATA_DIR / "ball_by_ball.csv") 
    print("\n Schema Audit Complete.")