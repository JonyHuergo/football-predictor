import joblib
import pandas as pd

# Load model and upcoming fixtures
model = joblib.load("./models/model.pkl")
fixtures = pd.read_csv("./data/fixtures/next_matchday.csv")

# Predict probabilities
fixtures['Prob_H'] = model.predict_proba(fixtures[features])[:, 0]  # Home win prob
fixtures['EV'] = (fixtures['Prob_H'] * fixtures['B365H']) - 1       # Expected Value

# Save value bets
value_bets = fixtures[fixtures['EV'] > 0]
value_bets.to_csv("./data/value_bets.csv")