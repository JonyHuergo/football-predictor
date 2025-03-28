# Preprocess each season with its corresponding FIFA ratings
python scripts/preprocess.py \
    --input_path "data/raw/E1_2021_2022.csv" \
    --output_path "data/processed/E1_2021_2022.feather" \
    --fifa_ratings_path "data/raw/FIFA_2022_RATINGS.csv"

python scripts/preprocess.py \
    --input_path "data/raw/E1_2022_2023.csv" \
    --output_path "data/processed/E1_2022_2023.feather" \
    --fifa_ratings_path "data/raw/FIFA_2023_RATINGS.csv"

python scripts/preprocess.py \
    --input_path "data/raw/E1_2023_2024.csv" \
    --output_path "data/processed/E1_2023_2024.feather" \
    --fifa_ratings_path "data/raw/FIFA_2024_RATINGS.csv"

python scripts/preprocess.py \
    --input_path "data/raw/E1_2024_2025.csv" \
    --output_path "data/processed/E1_2024_2025.feather" \
    --fifa_ratings_path "data/raw/FIFA_2025_RATINGS.csv"