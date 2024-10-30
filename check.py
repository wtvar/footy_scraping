import pandas as pd


def inspect_data():
    # Load processed data
    processed_df = pd.read_csv('processed_data.csv')
    
    print("\nProcessed Data Info:")
    print("Shape:", processed_df.shape)
    print("\nColumns:", processed_df.columns.tolist())
    
    if 'team' in processed_df.columns:
        print("\nSample teams:", processed_df['team'].unique()[:5])
    if 'home_team' in processed_df.columns:
        print("\nSample home teams:", processed_df['home_team'].unique()[:5])
        
    print("\nFirst few rows:")
    print(processed_df.head())

if __name__ == "__main__":
    inspect_data()
