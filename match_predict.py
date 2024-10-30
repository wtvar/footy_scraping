import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from scipy import stats

def calculate_gameweek(match_date, df):
    """
    Calculate gameweek for a given date based on existing data patterns
    """
    # Get the closest previous match date and its gameweek
    previous_matches = df[df['date'] < match_date].sort_values('date', ascending=False)
    if not previous_matches.empty:
        return previous_matches.iloc[0]['gameweek']
    return 1  # Default to 1 if no previous matches

def predict_match_outcome(home_team, away_team, match_date, df, model):
    """
    Predict match outcome probabilities for a given fixture
    """
    if isinstance(match_date, str):
        match_date = datetime.strptime(match_date, '%Y-%m-%d')
    
    # Get recent matches before the match_date
    recent_matches = df[df['date'] < match_date].copy()
    
    try:
        # Get last matches for home team
        home_recent = recent_matches[
            (recent_matches['team'] == home_team) & 
            (recent_matches['is_home'] == True)
        ].sort_values('date', ascending=False).head(8)
        
        if home_recent.empty:
            print(f"No recent home matches found for {home_team}")
            print(f"Available teams: {sorted(df['team'].unique().tolist())}")
            return None
        
        # Get last matches for away team
        away_recent = recent_matches[
            (recent_matches['team'] == away_team) & 
            (recent_matches['is_home'] == False)
        ].sort_values('date', ascending=False).head(8)
        
        if away_recent.empty:
            print(f"No recent away matches found for {away_team}")
            return None
        
        # Calculate gameweek for the prediction date
        gameweek = calculate_gameweek(match_date, df)
        
        # Get all feature columns (excluding certain columns)
        exclude_cols = ['date', 'team', 'is_home', 'target_xg', 'target_np_xg']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create feature vector using the most recent values
        match_features = {'gameweek': gameweek}  # Include gameweek
        for col in feature_cols:
            if col == 'gameweek':
                continue  # Already added
            elif col.startswith('home_'):
                match_features[col] = home_recent[col].iloc[0]
            elif col.startswith('away_'):
                match_features[col] = away_recent[col].iloc[0]
            elif col.startswith('relative_'):
                match_features[col] = home_recent[col].iloc[0]  # Use home team's relative metrics
                
    except Exception as e:
        print(f"Error calculating features: {str(e)}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Create feature vector
    X = pd.DataFrame([match_features])
    
    # Make prediction
    predicted_xg = model.predict(X)[0]
    
    # Calculate win/draw/loss probabilities using Poisson distribution
    home_goals_prob = pd.Series(stats.poisson.pmf(k=np.arange(0, 10), mu=predicted_xg))
    away_goals_prob = pd.Series(stats.poisson.pmf(k=np.arange(0, 10), mu=predicted_xg * 0.8))
    
    win_prob = 0
    draw_prob = 0
    lose_prob = 0
    
    for i in range(10):
        for j in range(10):
            p = home_goals_prob[i] * away_goals_prob[j]
            if i > j:
                win_prob += p
            elif i == j:
                draw_prob += p
            else:
                lose_prob += p
    
    # Get recent form information
    home_form = home_recent['home_form'].iloc[0]
    away_form = away_recent['away_form'].iloc[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'match_date': match_date.strftime('%Y-%m-%d'),
        'gameweek': gameweek,
        'predicted_home_xg': predicted_xg,
        'predicted_away_xg': predicted_xg * 0.8,
        'win_probability': win_prob,
        'draw_probability': draw_prob,
        'lose_probability': lose_prob,
        'home_form': home_form,
        'away_form': away_form,
        'key_features': {
            'home_ppda': match_features['home_rolling_ppda_4'],
            'away_ppda': match_features['away_rolling_ppda_7'],
            'home_deep_completions': match_features['home_rolling_deep_completions_5'],
            'relative_ppda': match_features['relative_ppda'],
            'relative_deep_completions': match_features['relative_deep_completions']
        }
    }

def main():
    # Load processed data
    try:
        df = pd.read_csv('processed_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        print("Data loaded successfully")
        print(f"Available teams: {sorted(df['team'].unique().tolist())}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    try:
        # Try to load the model from the model_outputs directory
        model = joblib.load('model_outputs/best_model.pkl')
        print("Model loaded successfully")
    except FileNotFoundError:
        try:
            # Try to load from current directory if not found in model_outputs
            model = joblib.load('best_model.pkl')
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Error: Could not find the trained model file 'best_model.pkl'")
            print("Please ensure you have run model_development.py first to train and save the model.")
            return

    # Example prediction
    prediction = predict_match_outcome(
        home_team='Newcastle United',
        away_team='Arsenal',
        match_date='2024-11-02',
        df=df,
        model=model
    )
    
    if prediction:
        # Print results
        print(f"\nPrediction for {prediction['home_team']} vs {prediction['away_team']}")
        print(f"Date: {prediction['match_date']} (Gameweek: {prediction['gameweek']})")
        print(f"\nPredicted xG:")
        print(f"Home: {prediction['predicted_home_xg']:.2f}")
        print(f"Away: {prediction['predicted_away_xg']:.2f}")
        print(f"\nMatch Outcome Probabilities:")
        print(f"Home Win: {prediction['win_probability']:.1%}")
        print(f"Draw: {prediction['draw_probability']:.1%}")
        print(f"Away Win: {prediction['lose_probability']:.1%}")
        print(f"\nTeam Form:")
        print(f"Home Form: {prediction['home_form']:.2f}")
        print(f"Away Form: {prediction['away_form']:.2f}")
        print(f"\nKey Features:")
        for feature, value in prediction['key_features'].items():
            print(f"{feature}: {value:.3f}")

if __name__ == "__main__":
    main()