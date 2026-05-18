import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import feedparser

# --- Configuration ---
DATA_DIR = "data"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(DATA_DIR, "movies.csv")
HYPERGRAPH_FILE = os.path.join(DATA_DIR, "hypergraph.json")

RETRIEVER_MODEL = 'all-MiniLM-L6-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class HypergraphAgent:
    def __init__(self):
        print("Initializing Logical Agent...")
        
        if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(METADATA_FILE) or not os.path.exists(HYPERGRAPH_FILE):
            raise FileNotFoundError("Required data files are missing. Please run the embedding and graph builder scripts.")
        
        self.embeddings = np.load(EMBEDDING_FILE)
        self.df = pd.read_csv(METADATA_FILE)

        with open(HYPERGRAPH_FILE, 'r') as f:
            self.hypergraph = json.load(f)

        print("   Loading fast retriever model...")
        self.retriever = SentenceTransformer(RETRIEVER_MODEL)

        print("   Loading cross-encoder reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        self.title_to_index = {title: idx for idx, title in enumerate(self.df['title'])}
        print("✅ Agent Ready.")


    def get_query_vector(self, query):
        return self.retriever.encode(query)
    
    def get_letterboxd_vector(self, username, mode="recent"):
        """
        Fetches user activity from Letterboxd.
        Modes:
          - 'recent': Uses the last 5 watched movies (regardless of rating).
          - 'favorites': Scans deep to find the last 5 movies rated 4 stars or higher.
        """
        rss_url = f"https://letterboxd.com/{username}/rss/"
        print(f"📡 Connecting to Letterboxd ({mode} mode): {rss_url}")
        
        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            print(f"❌ Network Error: {e}")
            return None
        
        if not feed.entries:
            print("❌ No entries found. Is the username correct?")
            return None

        vectors = []
        found_titles = []
        
        # CONFIGURATION SWITCHER
        if mode == "favorites":
            scan_depth = 100    # Look deeper to find the gems
            min_rating = 4    # Only 4/5 or 5/5 stars
            target_matches = 10
            print(f"   🔎 Scanning last {scan_depth} entries for ★★★★+ movies...")
        else:
            scan_depth = 20     # Look at recent history
            min_rating = 0.0    # Accept anything
            target_matches = 5
            print(f"   🔎 Scanning last {scan_depth} entries for recent activity...")

        
        entries_to_check = feed.entries[:scan_depth]

        for entry in entries_to_check:
            # 1. CHECK RATING (Only for 'favorites' mode)
            if mode == "favorites":
                rating = entry.get('letterboxd_memberrating', None)
                if rating is None or float(rating) < min_rating:
                    continue

            # 2. GET TITLE
            raw_title = entry.title
            clean_title = raw_title.split(",")[0].strip()
            
            # 3. MATCH WITH DB
            idx = self.find_start_node(clean_title)
            
            if idx is not None:
                if clean_title not in found_titles:
                    # Log the find
                    if mode == "favorites":
                        print(f"     ✅ Loved It ({entry.get('letterboxd_memberrating')}★): {clean_title}")
                    else:
                        print(f"     ✅ Found: {clean_title}")
                        
                    found_titles.append(clean_title)
                    vectors.append(self.embeddings[idx])
            
            if len(vectors) >= target_matches:
                print("   🎯 Target reached. Stopping scan.")
                break
        
        if not vectors:
            print(f"❌ Scanned {len(entries_to_check)} movies but found 0 matches.")
            return None

        print(f"   📊 Profile built from: {found_titles}")
        user_vibe_vector = np.mean(vectors, axis=0)
        
        return user_vibe_vector, found_titles
    
    def get_couple_vector(self, user1, user2, mode="recent"):
        """
        Combines two Letterboxd profiles into a single vibe vector.
        """

        print(f"❤️ Calculating 'Watch Party' vector for {user1} + {user2}...")

        v1, titles1 = self.get_letterboxd_vector(user1, mode=mode)
        v2, titles2 = self.get_letterboxd_vector(user2, mode=mode)

        if v1 is None or v2 is None:
            print("Could not build both profiles. Aborting.")

            return None, []
        

        couple_vector = (v1+v2) / 2.0

        combined_titles = list(set(titles1 + titles2))

        return couple_vector, combined_titles


    def find_start_node(self, title):
        for t, idx in self.title_to_index.items():
            if str(t).lower() == str(title).lower():
                return idx
        return None

    def logical_reranker(self, candidate_indices, user_query):
        # 1. Get text for candidates
        candidate_texts = self.df.iloc[candidate_indices]['combined_text'].fillna("").tolist()
        pairs = [[user_query, text] for text in candidate_texts]

        # 2. Get raw scores (logits) from AI
        logic_scores = self.reranker.predict(pairs)
        
        # 3. Sort high to low
        sorted_indices = np.argsort(logic_scores)[::-1]
        
        best_local_idx = sorted_indices[0]
        best_global_idx = candidate_indices[best_local_idx]
        
        # 4. Calculate Confidence (Sigmoid: Convert raw score to 0.0 - 1.0)
        best_score = logic_scores[best_local_idx]
        confidence = 1 / (1 + np.exp(-best_score)) 

        # Return BOTH the index AND the confidence
        return best_global_idx, confidence

    def step(self, current_node_idx, query_vector, user_query, visited_indices=set(), previous_context=None, vibe_mode=False, user_vectors=None,strategy="safe"):
        """
        The Core Decision Logic:
        1. Selects the best Theme (Hyperedge).
        2. Selects the best Movie (Node) using MMR (Relevance vs Diversity).
        3. Uses 'Vibe Mode' to skip slow logic for Letterboxd vectors.
        """
        # --- PHASE 1: Choose Context (Hyperedge) ---
        available_edges = [k for k,v in self.hypergraph.items() if current_node_idx in v['indices']]

        if not available_edges:
            return None, "Dead End", 0.0
        
        # Score edges based on vector similarity to user query
        edge_scores = []
        for edge in available_edges:
            centroid = np.array(self.hypergraph[edge]['centroid'])
            score = cosine_similarity([centroid], [query_vector])[0][0]
            
            # BOREDOM PENALTY: Don't pick the same theme twice in a row
            if edge == previous_context:
                score *= 0.5 
            
            edge_scores.append(score)

        chosen_edge = available_edges[np.argmax(edge_scores)]

        # --- PHASE 2: Choose Movie (Node) with MMR ---
        members = self.hypergraph[chosen_edge]['indices']
        member_embeddings = self.embeddings[members]
        
        # A. Calculate Relevance (How close is it to the Goal?)
        relevance_scores = cosine_similarity(member_embeddings, [query_vector]).flatten()

        if user_vectors:
            # If we have two users, adjust the Relevance Score
            vec_a, vec_b = user_vectors
            
            # Calculate individual scores for all candidates
            scores_a = cosine_similarity(member_embeddings, [vec_a]).flatten()
            scores_b = cosine_similarity(member_embeddings, [vec_b]).flatten()
            
            if strategy == "adventurous":
                # 🦁 ADVENTUROUS: "Maximum Pleasure"
                # If ONE person loves it, we consider it.
                social_scores = np.maximum(scores_a, scores_b)
                
                # We give it huge weight (80%) because we want highs, not averages.
                relevance_scores = social_scores
                
            else: 
                # 🛡️ SAFE (Default): "Least Misery"
                # We are limited by the unhappiest person.
                social_scores = np.minimum(scores_a, scores_b)
                relevance_scores = (social_scores * 0.7) + (relevance_scores * 0.3)
        
        # B. Calculate Redundancy (How similar is it to where we are standing?)
        # We want to move FORWARD, not stay in the same spot.
        current_movie_vec = self.embeddings[current_node_idx]
        redundancy_scores = cosine_similarity(member_embeddings, [current_movie_vec]).flatten()
        
        # C. MMR Formula: Score = (Lambda * Relevance) - ((1-Lambda) * Redundancy)
        # Lambda 0.7 means: "I care 70% about the goal, 30% about being fresh."
        mmr_lambda = 0.7
        mmr_scores = (mmr_lambda * relevance_scores) - ((1 - mmr_lambda) * redundancy_scores)

        # D. Visited Penalty (Don't go back)
        if len(members) > 1:
            for j, m_idx in enumerate(members):
                if m_idx in visited_indices:
                    mmr_scores[j] = -999.0

        # --- PHASE 3: Select Winner (Vibe Mode vs Logic Mode) ---
        
        if vibe_mode:
            # VIBE MODE: Trust the MMR score implicitly (Pure Math)
            best_local_idx = np.argmax(mmr_scores)
            next_node_idx = members[best_local_idx]
            
            # For confidence, we still report the raw Relevance (Cosine Similarity)
            # We normalize it (0.0 to 1.0) so it looks nice in the UI
            raw_sim = relevance_scores[best_local_idx]
            confidence = min(max((raw_sim + 1) / 2, 0), 1)
            
            return next_node_idx, chosen_edge, confidence

        else:
            # TEXT MODE: Use Cross-Encoder to "Read" the top candidates
            # We take the top 15 survivors from the MMR filter
            top_k = min(15, len(members))
            top_local_indices = np.argsort(mmr_scores)[-top_k:]
            top_global_indices = [members[j] for j in top_local_indices]

            next_node_idx, confidence = self.logical_reranker(top_global_indices, user_query)

            return next_node_idx, chosen_edge, confidence
    
    def walk(self, start_movie, user_query, steps=5, precomputed_vector=None, blocklist=None, user_vectors=None,strategy="safe"):
        current_idx = self.find_start_node(start_movie)
        if current_idx is None:
            print(f"❌ Start movie '{start_movie}' not found. Defaulting to index 0.")
            current_idx = 0

        # 1. DETECT MODE
        vibe_mode = False
        if precomputed_vector is not None:
            query_vector = precomputed_vector
            vibe_mode = True
        else:
            query_vector = self.get_query_vector(user_query)

        # 2. INITIALIZE MEMORY (Visited Set)
        visited_indices = {current_idx}
        
        # --- THE FIX: ADD BLOCKLIST TO MEMORY ---
        if blocklist:
            print(f"   🚫 Blocking {len(blocklist)} movies from results...")
            for title in blocklist:
                idx = self.find_start_node(title)
                if idx is not None:
                    visited_indices.add(idx)

        start_title = self.df.iloc[current_idx]['title']
        display_query = user_query if user_query else "User Persona Match"
        print(f"\n🚶 Starting walk at: {start_title} -> Goal: '{display_query}' (Vibe Mode: {vibe_mode})")
        
        path = []
        path.append({"step": 0, "movie": start_title, "context": "START", "confidence": "N/A"})

        previous_confidence = 0.0
        previous_context = None

        for i in range(steps):
            next_idx, context, confidence = self.step(
                current_idx, 
                query_vector, 
                user_query, 
                visited_indices, 
                previous_context, 
                vibe_mode=vibe_mode,
                user_vectors=user_vectors,
                strategy=strategy
            )

            if next_idx is None or next_idx in visited_indices:
                print("   (Agent is stuck)")
                break

            # Peak Detection
            if i > 0: 
                drop_ratio = confidence / previous_confidence if previous_confidence > 0 else 0
                threshold = 0.60 if vibe_mode else 0.15
                
                if confidence < threshold:
                     if not vibe_mode: 
                        print(f"   🛑 Stopping: Confidence ({confidence:.1%}) is too low.")
                        break

            current_idx = next_idx
            visited_indices.add(current_idx)
            previous_confidence = confidence
            previous_context = context
            
            movie_title = self.df.iloc[current_idx]['title']
            score_display = f"{confidence:.1%}"
            
            print(f"   Step {i+1}: [{context}] -> {movie_title} (Confidence: {score_display})")
            
            path.append({
                "step": i+1, "movie": movie_title, 
                "context": context, "confidence": score_display
            })

        return path

if __name__ == "__main__":
    agent = HypergraphAgent()
    
    # Test 1
    agent.walk("Inception", "I want a dark psychological thriller", steps=3)
    
    print("-" * 40)
    
    # Test 2
    agent.walk("The Dark Knight", "I want a romantic comedy", steps=3)