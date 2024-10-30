import os
from dotenv import load_dotenv
import soccerdata as sd
import pandas as pd
from datetime import datetime

# Load the .env file
load_dotenv()

def get_current_season():
    current_year = datetime.now().year
    if datetime.now().month > 7:  # Assuming the season starts in August
        return f"{current_year}/{current_year+1}"
    else:
        return f"{current_year-1}/{current_year}"

def initial_data_load():
    seasons = [f"{year}/{year+1}" for year in range(2014, int(get_current_season().split('/')[0]))]
    understat = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    team_match_stats = understat.read_team_match_stats()
    
    # Save the full dataset
    team_match_stats.to_csv('full_dataset.csv', index=False)
    print(f"Full dataset saved with shape: {team_match_stats.shape}")
    
    return team_match_stats

def update_data():
    current_season = get_current_season()
    understat = sd.Understat(leagues="ENG-Premier League", seasons=current_season)
    new_data = understat.read_team_match_stats()
    
    # Load existing data
    if os.path.exists('full_dataset.csv'):
        existing_data = pd.read_csv('full_dataset.csv')
        print(f"Existing data shape: {existing_data.shape}")
        
        # Identify new matches
        existing_matches = set(existing_data['game_id'])
        new_matches = new_data[~new_data['game_id'].isin(existing_matches)]
        
        if not new_matches.empty:
            # Append new matches to existing data
            updated_data = pd.concat([existing_data, new_matches], ignore_index=True)
            updated_data.to_csv('full_dataset.csv', index=False)
            print(f"Added {len(new_matches)} new matches. Updated data shape: {updated_data.shape}")
        else:
            print("No new matches to add.")
    else:
        print("No existing dataset found. Creating new dataset with current season data.")
        new_data.to_csv('full_dataset.csv', index=False)
        print(f"New dataset created with shape: {new_data.shape}")

def main():
    if not os.path.exists('full_dataset.csv'):
        print("Initial data load...")
        initial_data_load()
    else:
        print("Updating data...")
        update_data()

if __name__ == "__main__":
    main()
