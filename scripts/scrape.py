import pandas as pd
import requests
from bs4 import BeautifulSoup

# Example: Fetch Championship data from football-data.co.uk
def scrape_championship_data():
    url = "https://www.football-data.co.uk/mmz4281/2324/E1.csv"
    df = pd.read_csv(url)
    df.to_csv("./data/raw/championship_2023_24.csv", index=False)

# Run daily via GitHub Actions or cron job
if __name__ == "__main__":
    scrape_championship_data()