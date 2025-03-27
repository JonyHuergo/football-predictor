import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FootballDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.required_columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
            'HS', 'AS', 'HST', 'AST', 'B365H', 'B365D', 'B365A'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Clean raw data
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.sort_values('Date', inplace=True)
        
        # Calculate points
        df['Home_Pts'] = np.where(df['FTR'] == 'H', 3, np.where(df['FTR'] == 'D', 1, 0))
        df['Away_Pts'] = np.where(df['FTR'] == 'A', 3, np.where(df['FTR'] == 'D', 1, 0))
        
        # Create rolling stats (5 games)
        def calculate_rolling_stats(group, stat_cols, prefix):
            rolling_stats = group[stat_cols].rolling(5, min_periods=1).mean()
            return rolling_stats.add_prefix(prefix)
        
        # Home team stats
        home_groups = df.groupby('HomeTeam')
        home_stats = []
        for team, group in home_groups:
            stats = calculate_rolling_stats(group, ['FTHG', 'FTAG', 'HST', 'AST', 'Home_Pts'], 'Home_')
            stats['HomeTeam'] = team
            stats['Date'] = group['Date']
            home_stats.append(stats)
        
        home_stats = pd.concat(home_stats)
        
        # Away team stats
        away_groups = df.groupby('AwayTeam')
        away_stats = []
        for team, group in away_groups:
            stats = calculate_rolling_stats(group, ['FTAG', 'FTHG', 'AST', 'HST', 'Away_Pts'], 'Away_')
            stats['AwayTeam'] = team
            stats['Date'] = group['Date']
            away_stats.append(stats)
        
        away_stats = pd.concat(away_stats)
        
        # Merge stats back
        df = pd.merge(df, home_stats, on=['HomeTeam', 'Date'], how='left')
        df = pd.merge(df, away_stats, on=['AwayTeam', 'Date'], how='left')
        
        # Create differential features
        df['Diff_Goals'] = df['Home_FTHG'] - df['Away_FTAG']
        df['Diff_Shots_Target'] = df['Home_HST'] - df['Away_AST']
        df['Diff_Points'] = df['Home_Home_Pts'] - df['Away_Away_Pts']
        
        # Odds features
        df['Avg_Odds_Home_Prob'] = 1 / df['B365H']
        df['Avg_Odds_Draw_Prob'] = 1 / df['B365D']
        df['Avg_Odds_Away_Prob'] = 1 / df['B365A']
        
        # Last 3 results
        df['Home_Win'] = (df['FTR'] == 'H').astype(int)
        df['Away_Loss'] = (df['FTR'] == 'A').astype(int)
        
        home_last3 = df.groupby('HomeTeam')['Home_Win'].rolling(3, min_periods=1).mean()
        home_last3 = home_last3.reset_index(level=0, drop=True)
        
        away_last3 = df.groupby('AwayTeam')['Away_Loss'].rolling(3, min_periods=1).mean()
        away_last3 = away_last3.reset_index(level=0, drop=True)
        
        df['Home_Last3_Wins'] = home_last3
        df['Away_Last3_Losses'] = away_last3
        
        # Select final features
        features = [
            'Diff_Goals',
            'Diff_Points',
            'Diff_Shots_Target',
            'Avg_Odds_Home_Prob',
            'Home_Last3_Wins',
            'Away_Last3_Losses'
        ]
        
        return df[features].fillna(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path, encoding='utf-8-sig')
    processed_df = FootballDataPreprocessor().transform(df)
    processed_df.to_feather(args.output_path)