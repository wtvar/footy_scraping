import pandas as pd
import numpy as np

full_data_name = "full_dataset.csv"



def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def preprocess_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'home_team'])
    
    # Add gameweek column
    df['gameweek'] = (df.groupby(df['date'].dt.to_period('W'))
                        .ngroup() + 1)
    
    home_columns = ['date', 'gameweek', 'home_team', 'home_goals', 'home_xg', 'home_np_xg', 'home_ppda', 'home_deep_completions']
    away_columns = ['date', 'gameweek', 'away_team', 'away_goals', 'away_xg', 'away_np_xg', 'away_ppda', 'away_deep_completions']
    
    home_df = df[home_columns].copy()
    away_df = df[away_columns].copy()
    
    new_columns = ['date', 'gameweek', 'team', 'goals', 'xg', 'np_xg', 'ppda', 'deep_completions']
    home_df.columns = new_columns
    away_df.columns = new_columns
    
    home_df['is_home'] = True
    away_df['is_home'] = False
    
    team_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    print(f"Preprocessed data shape: {team_df.shape}")
    return team_df

def engineer_features(df, window_sizes=[4, 5, 6,7,8]):
    df = df.copy()
    df = df.sort_values(['team', 'date'])
    features = ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']
    
    for window in window_sizes:
        for feature in features:
            for is_home in [True, False]:
                prefix = 'home' if is_home else 'away'
                
                mask = df['is_home'] == is_home
                
                rolled = df[mask].groupby('team')[feature].shift().rolling(
                    window=window, 
                    min_periods=1
                ).mean().reset_index(level=0, drop=True)
                
                df.loc[mask, f'{prefix}_rolling_{feature}_{window}'] = rolled
    
    for is_home in [True, False]:
        prefix = 'home' if is_home else 'away'
        mask = df['is_home'] == is_home
        
        form = df[mask].groupby('team')['goals'].shift().rolling(
            window=5, 
            min_periods=1
        ).apply(lambda x: (x > x.shift(1)).sum()).reset_index(level=0, drop=True)
        
        df.loc[mask, f'{prefix}_form'] = form
    
    print(f"Data shape after feature engineering: {df.shape}")
    print(f"Number of NaN values: {df.isna().sum().sum()}")
    
    df = df.fillna(0)
    
    print(f"Final data shape: {df.shape}")
    return df

def prepare_for_modeling(df):
    df = df.copy()
    feature_cols = [col for col in df.columns if col.startswith(('home_rolling', 'away_rolling', 'home_form', 'away_form', 'relative_'))]
    
    modeling_df = df[['date', 'gameweek', 'team', 'is_home'] + feature_cols].copy()
    modeling_df['target_xg'] = df['xg']
    modeling_df['target_np_xg'] = df['np_xg']
    
    print(f"Modeling data shape: {modeling_df.shape}")
    return modeling_df


def calculate_relative_strength(df, window_size=5):
    print("Starting calculate_relative_strength function...")
    df = df.copy()
    
    print("Calculating league averages...")
    metrics = ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']
    league_avg = df.groupby('date')[metrics].mean()
    
    print("Calculating team averages...")
    # Calculate team averages over the window
    team_avg = df.sort_values(['team', 'date']).set_index('date').groupby('team')[metrics].rolling(window=window_size, min_periods=1).mean().reset_index()
    
    # Rename columns
    team_avg = team_avg.rename(columns={col: f'{col}_avg' for col in metrics})
    
    print("Team average columns:")
    print(team_avg.columns)
    print("\nSample of team averages:")
    print(team_avg.head())
    
    print("Merging team averages...")
    # Merge team averages back to the original dataframe
    df = df.merge(team_avg, on=['team', 'date'], how='left')
    
    print("Calculating relative strength...")
    # Calculate relative strength compared to league average
    for metric in metrics:
        df[f'relative_{metric}'] = df[f'{metric}_avg'] / league_avg.loc[df['date'], metric].values
    
    print("Finished calculate_relative_strength function.")
    return df




def add_opponent_strength(df):
    print("Starting add_opponent_strength function...")
    df = df.copy()
    
    print("Creating match opponents mapping...")
    # Create a mapping of each match to the opponent's team
    match_opponents = df[['date', 'team', 'is_home']].copy()
    match_opponents['opponent'] = df.groupby('date')['team'].transform(lambda x: x.iloc[::-1].values)
    
    print("Merging opponent's relative strength...")
    # Merge opponent's relative strength to the main dataframe
    opponent_strength = df[['date', 'team'] + [col for col in df.columns if col.startswith('relative_')]]
    opponent_strength = opponent_strength.rename(columns={col: f'opponent_{col}' for col in opponent_strength.columns if col.startswith('relative_')})
    
    df = df.merge(match_opponents, on=['date', 'team', 'is_home'], how='left')
    df = df.merge(opponent_strength, left_on=['date', 'opponent'], right_on=['date', 'team'], how='left', suffixes=('', '_opponent'))
    
    print("Calculating rolling average of opponent strength...")
    # Calculate rolling average of opponent strength
    opponent_cols = [col for col in df.columns if col.startswith('opponent_relative_')]
    for col in opponent_cols:
        df[f'{col}_rolling_5'] = df.sort_values('date').groupby('team')[col].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    print("Finished add_opponent_strength function.")
    return df

# Modify the main function to include these new steps
def main():
    df = load_data(full_data_name)
    team_df = preprocess_data(df)
    featured_df = engineer_features(team_df)
    
    print("Columns in featured_df:")
    print(featured_df.columns)
    print("\nSample data:")
    print(featured_df[['team', 'date', 'xg', 'np_xg', 'goals', 'ppda', 'deep_completions']].head())
    
    print("Starting relative strength calculations...")
    featured_df = calculate_relative_strength(featured_df)
    
    modeling_df = prepare_for_modeling(featured_df)
    
    modeling_df.to_csv('processed_data.csv', index=False)
    print("Data preprocessing and feature engineering completed. Processed data saved to 'processed_data.csv'.")
    print(f"Final CSV shape: {modeling_df.shape}")
    print(f"Columns in final CSV: {', '.join(modeling_df.columns)}")
    
    print("\nSample of processed data:")
    print(modeling_df[['team', 'date', 'relative_xg', 'relative_np_xg', 'relative_goals', 'relative_ppda', 'relative_deep_completions']].head())


if __name__ == "__main__":
    main()