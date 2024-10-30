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

def engineer_features(df, window_sizes=[4, 5, 6, 7, 8]):
    df = df.copy()
    df = df.sort_values(['team', 'date'])
    
    # First, create opponent columns
    print("Creating opponent columns...")
    opponent_cols = {}
    for col in ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']:
        opponent_cols[f'opponent_{col}'] = df.groupby('date')[col].transform(lambda x: x.iloc[::-1].values)
    
    # Create a new dataframe with all columns we'll need
    new_data = {
        **opponent_cols,  # Add opponent columns first
    }
    
    # Pre-create all rolling average columns
    print("Pre-creating rolling average columns...")
    for window in window_sizes:
        for feature in ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']:
            # Scoring columns
            new_data[f'home_rolling_{feature}_{window}'] = np.nan
            new_data[f'away_rolling_{feature}_{window}'] = np.nan
            new_data[f'rolling_{feature}_{window}'] = np.nan
            
            # Conceding columns
            new_data[f'home_rolling_{feature}_conceded_{window}'] = np.nan
            new_data[f'away_rolling_{feature}_conceded_{window}'] = np.nan
            new_data[f'rolling_{feature}_conceded_{window}'] = np.nan
    
    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)
    
    # Calculate rolling averages
    print("\nCalculating rolling averages...")
    for window in window_sizes:
        print(f"Window size: {window}")
        for feature in ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']:
            # Calculate home stats
            home_mask = df['is_home']
            home_values = df.loc[home_mask, feature]
            home_rolling = home_values.groupby(df.loc[home_mask, 'team']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Calculate away stats
            away_mask = ~df['is_home']
            away_values = df.loc[away_mask, feature]
            away_rolling = away_values.groupby(df.loc[away_mask, 'team']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Calculate conceded stats (using opponent columns)
            home_conceded = df.loc[home_mask, f'opponent_{feature}']
            home_conceded_rolling = home_conceded.groupby(df.loc[home_mask, 'team']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            away_conceded = df.loc[away_mask, f'opponent_{feature}']
            away_conceded_rolling = away_conceded.groupby(df.loc[away_mask, 'team']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Assign all values at once for each feature
            df.loc[home_mask, f'home_rolling_{feature}_{window}'] = home_rolling
            df.loc[away_mask, f'away_rolling_{feature}_{window}'] = away_rolling
            df.loc[home_mask, f'home_rolling_{feature}_conceded_{window}'] = home_conceded_rolling
            df.loc[away_mask, f'away_rolling_{feature}_conceded_{window}'] = away_conceded_rolling
            
            # Create consolidated columns
            df[f'rolling_{feature}_{window}'] = np.where(
                df['is_home'],
                df[f'home_rolling_{feature}_{window}'],
                df[f'away_rolling_{feature}_{window}']
            )
            
            df[f'rolling_{feature}_conceded_{window}'] = np.where(
                df['is_home'],
                df[f'home_rolling_{feature}_conceded_{window}'],
                df[f'away_rolling_{feature}_conceded_{window}']
            )
    
    # Print sample to verify
    print("\nSample of consolidated rolling averages for Arsenal:")
    sample_mask = (df['team'] == 'Arsenal') & (df['date'] < '2014-12-01')
    sample_cols = ['date', 'is_home', 'xg', 'opponent_xg',
                  'rolling_xg_4', 'rolling_xg_conceded_4']
    print(df[sample_mask][sample_cols].head(10))
    
    return df

def prepare_for_modeling(df):
    df = df.copy()
    
    # Combine home/away columns into single features
    for window in [4, 5, 6, 7, 8]:
        for metric in ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']:
            # Combine scoring metrics
            df[f'rolling_{metric}_{window}'] = np.where(
                df['is_home'],
                df[f'home_rolling_{metric}_{window}'],
                df[f'away_rolling_{metric}_{window}']
            )
            
            # Combine conceded metrics
            df[f'rolling_{metric}_conceded_{window}'] = np.where(
                df['is_home'],
                df[f'home_rolling_{metric}_conceded_{window}'],
                df[f'away_rolling_{metric}_conceded_{window}']
            )
    
    # Select only the combined columns and other relevant features
    feature_cols = [
        col for col in df.columns if any([
            col.startswith('rolling_'),
            col.startswith('relative_'),
            col.startswith('offensive_'),
            col.startswith('defensive_'),
            col.startswith('xg_over')
        ])
    ]
    
    modeling_df = df[['date', 'gameweek', 'team', 'is_home'] + feature_cols].copy()
    modeling_df['target_xg'] = df['xg']
    modeling_df['target_np_xg'] = df['np_xg']
    
    print(f"Modeling data shape: {modeling_df.shape}")
    print("\nSample of final features:")
    print(modeling_df[['rolling_xg_4', 'rolling_xg_conceded_4', 'target_xg']].head())
    print("\nMissing values in final dataset:")
    print(modeling_df.isnull().sum().sum())
    
    return modeling_df


def calculate_relative_strength(df):
    df = df.copy()
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    
    # Calculate relative metrics with safety checks
    metrics = ['xg', 'np_xg', 'goals', 'ppda', 'deep_completions']
    for metric in metrics:
        opponent_metric = f'opponent_{metric}'
        relative_metric = f'relative_{metric}'
        
        # Handle division by zero and inf values
        df[relative_metric] = np.where(
            df[opponent_metric] == 0,
            df[metric] / epsilon,  # when opponent metric is 0
            df[metric] / (df[opponent_metric] + epsilon)  # normal case
        )
        
        # Cap extreme values
        df[relative_metric] = df[relative_metric].clip(0, 5)  # adjust max as needed
    
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

def add_advanced_metrics(df):
    df = df.copy()
    
    # Calculate offensive efficiency
    df['offensive_efficiency'] = (
        df.groupby('team')['goals'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        ) / 
        df.groupby('team')['xg'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
    )
    
    # Calculate defensive efficiency
    df['defensive_efficiency'] = (
        df.groupby('team')['opponent_goals'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        ) / 
        df.groupby('team')['opponent_xg'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
    )
    
    # Calculate xG overperformance
    df['xg_overperformance'] = (
        df.groupby('team')['goals'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        ) - 
        df.groupby('team')['xg'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
    )
    
    return df

def calculate_rolling_averages(df, window_size):
    # Add print statements for debugging
    print(f"\nCalculating {window_size}-game rolling averages:")
    print(f"Sample of xG values before rolling: {df['xg'].head()}")
    
    rolling_stats = df.groupby('team').rolling(window=window_size, min_periods=1)['xg'].mean()
    
    print(f"Sample of rolling xG values: {rolling_stats.head()}")
    return rolling_stats

def prepare_features(df):
    df = df.copy()
    
    # Log transform xG features (adding small constant to handle zeros)
    xg_columns = [col for col in df.columns if 'xg' in col.lower()]
    for col in xg_columns:
        df[f'{col}_log'] = np.log1p(df[col])
    
    return df

def add_interaction_features(df):
    df = df.copy()
    
    # Create interaction between home and away xG
    df['home_away_xg_interaction'] = df['home_rolling_xg_4'] * df['away_rolling_xg_4']
    
    # Create relative strength features
    df['xg_dominance'] = df['home_rolling_xg_4'] - df['away_rolling_xg_4']
    
    return df

def check_data_quality(df):
    print("\n=== Data Quality Report ===\n")
    
    # 1. Basic Dataset Info
    print(f"Dataset Shape: {df.shape}")
    print(f"Time Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique teams: {df['team'].nunique()}")
    
    # 2. Missing Values Analysis
    missing_vals = df.isnull().sum()
    missing_percentages = (missing_vals / len(df)) * 100
    print("\nColumns with missing values:")
    missing_report = pd.DataFrame({
        'Missing Values': missing_vals,
        'Percentage': missing_percentages
    })
    print(missing_report[missing_report['Missing Values'] > 0])
    
    # 3. Check for Extreme Values in Key Metrics
    key_metrics = ['relative_xg', 'relative_goals', 'offensive_efficiency', 'defensive_efficiency']
    print("\nExtreme Values Analysis:")
    for metric in key_metrics:
        if metric in df.columns:
            q1 = df[metric].quantile(0.25)
            q3 = df[metric].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
            
            print(f"\n{metric}:")
            print(f"Range: {df[metric].min():.2f} to {df[metric].max():.2f}")
            print(f"Outliers: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}%)")
            if len(outliers) > 0:
                print("Sample outliers:")
                print(outliers[['date', 'team', metric]].head())
    
    # 4. Check Rolling Average Consistency
    print("\nRolling Average Consistency Check:")
    window_sizes = [4, 5, 6, 7, 8]
    for size in window_sizes:
        col = f'rolling_xg_{size}'
        
        # Check if values are within reasonable range
        unreasonable = df[df[col] > 5]
        if len(unreasonable) > 0:
            print(f"\nUnusual values for {size}-game window:")
            print(unreasonable[['date', 'team', col]].head())
    
    # 5. Team Frequency Analysis
    games_per_team = df['team'].value_counts()
    print("\nGames per team distribution:")
    print(f"Min games: {games_per_team.min()}")
    print(f"Max games: {games_per_team.max()}")
    if games_per_team.max() - games_per_team.min() > 10:
        print("\nTeams with unusual number of games:")
        unusual_teams = games_per_team[abs(games_per_team - games_per_team.mean()) > games_per_team.std()]
        print(unusual_teams)

# Modify the main function to include these new steps
def main():
    # Load data
    df = load_data(full_data_name)
    
    # Basic preprocessing
    team_df = preprocess_data(df)
    
    # Feature engineering steps
    print("Starting feature engineering...")
    featured_df = engineer_features(team_df)
    
    print("Calculating relative strength...")
    featured_df = calculate_relative_strength(featured_df)
    
    print("Adding advanced metrics...")
    featured_df = add_advanced_metrics(featured_df)
    
    # Prepare final modeling dataset
    modeling_df = prepare_for_modeling(featured_df)
    
    # Save processed data
    modeling_df.to_csv('processed_data.csv', index=False)
    print("Data preprocessing and feature engineering completed. Processed data saved to 'processed_data.csv'.")
    print(f"Final CSV shape: {modeling_df.shape}")
    print(f"Columns in final CSV: {', '.join(modeling_df.columns)}")
    
    print("\nSample of processed data:")
    print(modeling_df[['team', 'date', 'relative_xg', 'relative_np_xg', 
                      'offensive_efficiency', 'defensive_efficiency', 
                      'xg_overperformance', 'target_xg']].head())
    
    print("\nPerforming data quality checks...")
    check_data_quality(modeling_df)

if __name__ == "__main__":
    main()