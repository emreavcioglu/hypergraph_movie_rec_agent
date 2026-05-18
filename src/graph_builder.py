import json
import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "movies.csv")
HYPERGRAPH_FILE = os.path.join(DATA_DIR, "hypergraph.json")
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.npy")

# CONFIG: How big should the clusters be?
MAX_CLUSTER_SIZE = 50

def build_hypergraph():
    print("🏗️ Building Hypergraph from new Kaggle dataset...")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Missing {METADATA_FILE}. Run embedding_builder.py first!")
        
    df = pd.read_csv(METADATA_FILE)
    embeddings = np.load(EMBEDDING_FILE)
    
    hypergraph = {}

    print(f"   Processing {len(df)} movies...")

    # --- 1. BUILD GENRE EDGES ---
    # The new dataset uses simple strings: "Action, Adventure, Sci-Fi"
    
    genre_map = {} # Stores { "Action": [movie_idx_1, movie_idx_5], ... }

    for idx, row in df.iterrows():
        # Handle missing or non-string genres safely
        raw_genres = str(row.get('genres', ''))
        if raw_genres == 'nan': 
            raw_genres = ''
            
        # Split by comma
        current_genres = [g.strip() for g in raw_genres.split(',') if g.strip()]
        
        for g in current_genres:
            if g not in genre_map:
                genre_map[g] = []
            genre_map[g].append(idx)

    # Convert to Hypergraph Nodes
    for genre, indices in genre_map.items():
        # Optimization: Don't make clusters for obscure genres with only 1 movie
        if len(indices) < 5: 
            continue
            
        # Optimization: Limit massive genres (like 'Drama') to the most relevant/recent ones 
        # (Since our CSV is sorted by popularity, taking the first N is a good heuristic)
        limit_indices = indices[:MAX_CLUSTER_SIZE]
        
        # Calculate Centroid (Average Vector) for this Genre
        # This allows the Agent to "feel" what "Action" tastes like mathematically
        genre_vecs = embeddings[limit_indices]
        centroid = np.mean(genre_vecs, axis=0).tolist()

        node_name = f"Genre: {genre}"
        hypergraph[node_name] = {
            "indices": limit_indices,
            "centroid": centroid
        }
        print(f"   ✅ Created Edge: {node_name} ({len(limit_indices)} movies)")


    # --- 2. SAVE ---
    print(f"💾 Saving hypergraph with {len(hypergraph)} connections...")
    with open(HYPERGRAPH_FILE, 'w') as f:
        json.dump(hypergraph, f)
    
    print("✅ Hypergraph Ready.")

if __name__ == "__main__":
    build_hypergraph()