import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import kagglehub
from sklearn.preprocessing import normalize
import glob

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_DIR = "data"
EMBEDDING_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "movies.csv")

# Limit to top 20,000 movies to keep the agent fast and relevant
TOP_N_MOVIES = 20000 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Download Dataset via KaggleHub
    print("⬇️ Downloading latest TMDB dataset from Kaggle...")
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    print("Path to dataset files:", path)

    # Find the CSV file in the downloaded folder
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset!")
    
    raw_csv_path = csv_files[0]
    print(f"📖 Reading raw dataset: {raw_csv_path}")

    # 2. Load and Filter Data
    # We only read necessary columns to save memory
    df = pd.read_csv(raw_csv_path, usecols=['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count'])
    
    print(f"   Original count: {len(df)} movies")

    # Filter: Remove movies with no overview or title
    df.dropna(subset=['title', 'overview'], inplace=True)
    
    # Filter: Keep only movies with > 50 votes (removes obscure junk/test data)
    df = df[df['vote_count'] > 50].copy()
    
    # Sort: Most popular movies first
    df.sort_values(by='vote_count', ascending=False, inplace=True)
    
    # Truncate: Keep top N
    df = df.head(TOP_N_MOVIES)
    print(f"   ✅ Filtered down to top {len(df)} movies.")

    # 3. Process Text
    df['genres'] = df['genres'].fillna("Unknown")
    
    # Create the "Combined Text" for the AI to read
    df['combined_text'] = (
        "Title: " + df['title'].astype(str) + ". " +
        "Genre: " + df['genres'].astype(str) + ". " +
        "Overview: " + df['overview'].astype(str)
    )

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save the clean metadata for the App to use later
    df.to_csv(METADATA_FILE, index=False)
    print(f"✅ Saved metadata to {METADATA_FILE}")

    # 4. Generate Embeddings
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Encoding movie texts (This may take a few minutes)...")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True, device=device)

    # Convert to numpy and Normalize (L2)
    # This ensures Cosine Similarity math works correctly
    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings, norm='l2', axis=1)

    print(f"Saving embeddings with shape {embeddings.shape}")
    np.save(EMBEDDING_FILE, embeddings)
    print(f"✅ Saved embeddings to {EMBEDDING_FILE}")

if __name__ == "__main__":
    main()