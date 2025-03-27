/EFL-Championship-Betting
├── /data
│   ├── raw/                  # Store raw CSV/JSON from sources (e.g., football-data.co.uk)
│   ├── processed/            # Cleaned datasets (feather/parquet format for speed)
│   └── fixtures/             # Upcoming matches (update weekly)
├── /notebooks
│   ├── EDA.ipynb             # Exploratory data analysis
│   └── Model_Prototyping.ipynb  # Test models
├── /scripts
│   ├── scrape.py             # Web scraper for stats/odds
│   ├── preprocess.py         # Feature engineering pipeline
│   ├── train.py              # Model training script
│   └── predict.py            # Generate bets for next matchday
├── /models
│   ├── model.pkl             # Trained model (update weekly)
│   └── model_metrics.json    # Track accuracy/EV over time
├── .gitignore                # Ignore data/, .env, etc.
├── requirements.txt          # Python dependencies
└── README.md                 # Project docs (setup, goals, results)