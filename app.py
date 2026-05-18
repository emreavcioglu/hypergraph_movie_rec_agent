import sys
import json
import os
import networkx as nx
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QInputDialog, QMessageBox, QPushButton) # Added QPushButton
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Import your Agent
from src.agent import HypergraphAgent
from src.narrator import MovieNarrator

# --- Configuration ---
DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "movies.csv")
HYPERGRAPH_FILE = os.path.join(DATA_DIR, "hypergraph.json")
GROQ_API_KEY = "// Insert your GROQ API Key here//"

# VISUAL CONFIG: How many movies per cluster to draw?
# Increased to 30 for a richer "galaxy" feel.
MOVIES_PER_CLUSTER = 150

class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hypergraph Agent")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # 1. Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- HEADER LAYOUT WITH SEARCH BUTTON ---
        header_layout = QVBoxLayout()
        
        self.label = QLabel("Initializing Agent... (This may take a moment)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; color: #00d4ff;")
        header_layout.addWidget(self.label)

        # The Search Button
        self.search_btn = QPushButton("🔍 Search Start Node")
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc; color: white; border-radius: 5px; padding: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #005f9e; }
        """)
        self.search_btn.clicked.connect(self.search_movie)
        header_layout.addWidget(self.search_btn)
        
        layout.addLayout(header_layout)
        # ---------------------------------------------

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addPlot()
        self.view.setAspectLocked()
        self.view.hideAxis('bottom')
        self.view.hideAxis('left')
        layout.addWidget(self.graph_widget)

        # 2. Setup Data Layers
        self.lines_item = pg.PlotCurveItem(pen=pg.mkPen(color=(100, 100, 100, 50), width=1))
        self.view.addItem(self.lines_item)
        
        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter.sigClicked.connect(self.on_node_clicked)
        self.view.addItem(self.scatter)

        self.path_lines = pg.PlotCurveItem(pen=pg.mkPen(color='#00ff00', width=3))
        self.view.addItem(self.path_lines)

        self.pos_dict = {} 
        QTimer.singleShot(100, self.load_system)

    def load_system(self):
        try:
            self.agent = HypergraphAgent()
            self.narrator = MovieNarrator(api_key=GROQ_API_KEY)

        except Exception as e:
            self.label.setText(f"Error loading Agent: {e}")
            return

        self.build_graph_visuals()
        self.label.setText("System Ready. Click a dot or use Search to start.")

    def build_graph_visuals(self):
        print("🎨 Building Visualization...")
        # 1. Create a temporary graph for layout calculation
        G = nx.Graph()
        
        for theme_name, data in self.agent.hypergraph.items():
            G.add_node(theme_name, type='theme')
            indices = data['indices'][:MOVIES_PER_CLUSTER]
            
            for idx in indices:
                movie_id = f"M_{idx}"
                try:
                    title = self.agent.df.iloc[idx]['title']
                except:
                    title = "Unknown"
                
                G.add_node(movie_id, type='movie', title=title)
                G.add_edge(theme_name, movie_id)

        # 2. Calculate Layout
        self.pos_dict = nx.spring_layout(G, k=0.15, iterations=30, seed=42)
        
        spots = []
        connect_x = []
        connect_y = []

        # 3. Process Positions
        for node, (x, y) in self.pos_dict.items():
            node_type = G.nodes[node].get('type', 'movie')
            
            if node_type == 'theme':
                spots.append({'pos': (x, y), 'size': 20, 'brush': pg.mkBrush(255, 50, 50, 200), 'data': node})
                # Clean up display name
                short_name = node.replace("Theme: ", "").replace("Genre: ", "")
                text = pg.TextItem(short_name, anchor=(0.5, 0.5), color=(200, 200, 200))
                text.setPos(x, y)
                self.view.addItem(text)
            else:
                # Movie Nodes
                # Reduced size to 6 so it doesn't look too crowded with 30 movies/cluster
                spots.append({'pos': (x, y), 'size': 6, 'brush': pg.mkBrush(50, 150, 255, 150), 'data': node})

            # 4. Prepare Lines (Edges)
            for neighbor in G.neighbors(node):
                if neighbor in self.pos_dict:
                    n_x, n_y = self.pos_dict[neighbor]
                    connect_x.extend([x, n_x])
                    connect_y.extend([y, n_y])

        self.scatter.setData(spots)
        self.lines_item.setData(x=connect_x, y=connect_y, connect="pairs")
        print("✅ Visuals Ready")

    def search_movie(self):
        """Opens a text box to search for a movie by name."""
        title, ok = QInputDialog.getText(self, "Search", "Enter movie title to start from:")
        if ok and title:
            # Use the agent's lookup tool
            idx = self.agent.find_start_node(title)
            
            if idx is not None:
                real_title = self.agent.df.iloc[idx]['title']
                # Trigger the shared menu
                self.open_menu_for_movie(idx, real_title)
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find movie: '{title}'\n(Try a different spelling)")

    def on_node_clicked(self, plot_item, points):
        """Handle clicking on a visual dot."""
        clicked_point = points[0]
        node_id = clicked_point.data()
        
        if str(node_id).startswith("M_"):
            idx = int(node_id.split("_")[1])
            movie_title = self.agent.df.iloc[idx]['title']
            self.open_menu_for_movie(idx, movie_title)

    def open_menu_for_movie(self, idx, movie_title):
        """
        Shared Logic: Opens the action menu for a specific movie.
        """
        # Highlight the start node in Yellow so user sees it
        node_id = f"M_{idx}"
        if node_id in self.pos_dict:
            x, y = self.pos_dict[node_id]
            self.view.addItem(pg.ScatterPlotItem(pos=[(x, y)], size=20, brush=pg.mkBrush('y')))

        # --- MENU SELECTION ---
        items = ("Text Query", "Letterboxd Match","Watch Party (2 Users)")
        item, ok = QInputDialog.getItem(self, "Agent Mode", 
            f"Start at: {movie_title}\nChoose navigation mode:", items, 0, False)
        
        if ok and item:
            if item == "Text Query":
                query, ok_q = QInputDialog.getText(self, "Command", "Where to? (e.g. 'Funny space movie')")
                if ok_q:
                    self.run_agent_walk(movie_title, query_text=query)
                    
            elif item == "Letterboxd Match":
                modes = ("Recent Activity", "Favorites Only (4+ Stars)")
                mode_item, ok_m = QInputDialog.getItem(self, "Letterboxd Strategy", 
                    "Which movies should define the vibe?", modes, 0, False)
                
                if ok_m and mode_item:
                    # Convert UI text to Agent Code
                    lb_mode = "recent" if mode_item == "Recent Activity" else "favorites"
                    
                    username, ok_u = QInputDialog.getText(self, "Letterboxd", "Enter Username:")
                    if ok_u:
                        self.run_agent_walk(movie_title, letterboxd_user=username, lb_mode=lb_mode)

            elif item == "Watch Party (2 Users)":
                modes = ("Recent Activity", "Favorites Only (4+ Stars)")
                mode_item, ok_m = QInputDialog.getItem(self, "Letterboxd Strategy",
                    "Which movies should define the vibe?", modes, 0, False)
                
                if ok_m and mode_item:
                    lb_mode = "recent" if mode_item == "Recent Activity" else "favorites"
                    risk_levels = ("Safe Bet (No fighting)", "Adventurous (High highs, low lows)")
                    risk, ok_r = QInputDialog.getItem(self, "Risk Level", 
                    "How risky should the choice be?", risk_levels, 0, False)
                
                    if ok_r and risk:
                        # Map UI to code string
                        strategy = "safe" if "Safe" in risk else "adventurous"

                        username1, ok1 = QInputDialog.getText(self, "User 1", "Enter first Letterboxd username:")
                    if ok1 and username1:
                        username2, ok2 = QInputDialog.getText(self, "User 2", "Enter second Letterboxd username:")
                        if ok2:
                            self.run_watch_party(movie_title, username1, username2, lb_mode=lb_mode,strategy=strategy)    

    def run_agent_walk(self, start_movie, query_text=None, letterboxd_user=None, lb_mode="recent"):
        # --- DEBUG SPY ---
        print(f"🕵️ DEBUG APP: run_agent_walk called with lb_mode = '{lb_mode}'")
        # -----------------

        self.label.setText(f"🤖 Agent thinking...")
        QApplication.processEvents()
        
        query_vector = None
        user_intent = ""
        blocklist = [] 
        
        if letterboxd_user:
            self.label.setText(f"📡 Fetching Letterboxd data ({lb_mode} mode)...")
            QApplication.processEvents()
            
            # PASS THE MODE TO THE AGENT
            result = self.agent.get_letterboxd_vector(letterboxd_user, mode=lb_mode)

            if result is None:
                self.label.setText("Error: Could not fetch valid Letterboxd data.")
                QMessageBox.warning(self, "Error", "Could not find valid matches in your history.")
                return
            
            query_vector, found_titles = result
            
            # Add found movies to blocklist so we don't recommend them back to the user
            blocklist = found_titles 

            user_intent = f"Movies similar to {', '.join(found_titles)}"
            print(f"   🧠 Meaningful Intent generated: '{user_intent}'")
            
        elif query_text:
            user_intent = query_text
            # No vector needed here, walk() will generate it from text

        # 2. Run Agent
        # Passing blocklist prevents "Echo Chamber" recommendations
        path_data = self.agent.walk(
            start_movie, 
            user_intent, 
            steps=6, 
            precomputed_vector=query_vector,
            blocklist=blocklist
        )
        
        # 3. Draw Path
        path_x = []
        path_y = []
        full_text = f"Journey: {start_movie} ➡️ '{user_intent}'\n"
        
        for step in path_data:
            idx = self.agent.find_start_node(step['movie'])
            node_id = f"M_{idx}"
            
            if node_id in self.pos_dict:
                x, y = self.pos_dict[node_id]
                path_x.append(x)
                path_y.append(y)
            
            if step['step'] > 0:
                full_text += f"Step {step['step']}: {step['movie']} ({step['confidence']})\n"

        self.path_lines.setData(x=path_x, y=path_y)
        self.label.setText(f"✅ Arrived. Check popup for details.")
        QApplication.processEvents()
      
        try:
            if hasattr(self, 'narrator'):
                self.label.setText(f"🤖 Generating AI explanation...")
                QApplication.processEvents()
                narrative = self.narrator.explain_journey(start_movie, path_data, user_intent)
                full_text += f"\n\n🤖 AI Insight:\n{narrative}"
            else:
                print("⚠️ Narrator not initialized (Check load_system)")
        except Exception as e:
            print(f"⚠️ Narrator failed: {e}")
            full_text += "\n(AI Explanation unavailable)"
        # ----------------------------

        self.label.setText("✅ Done.")
        
        # POPUP IS NOW SAFE TO RUN
        msg = QMessageBox()
        msg.setWindowTitle("Agent Report")
        msg.setText(full_text)
        msg.exec()
        

    def run_watch_party(self, start_movie, username1, username2, lb_mode="recent",strategy="safe"):
        self.label.setText(f"Blending tastes for {username1} & {username2}...")
        QApplication.processEvents()
        
        vec1, _ = self.agent.get_letterboxd_vector(username1, mode=lb_mode)
        vec2, _ = self.agent.get_letterboxd_vector(username2, mode=lb_mode)

        query_vector, combined_titles = self.agent.get_couple_vector(username1, username2, mode=lb_mode)

        if query_vector is None:
            QMessageBox.warning(self, "Error", "Could not fetch data for one or both users.")
            self.label.setText("Error: Watch Party failed.")
            return
        
        user_intent = f"Watch party: {username1} + {username2}"

        path_data = self.agent.walk(
            start_movie,
            user_intent,
            steps=6,
            precomputed_vector=query_vector,
            blocklist=combined_titles,
            user_vectors=[vec1,vec2],
            strategy=strategy
        )

        path_x = []
        path_y = []
        full_text = f"Journey: {start_movie} ➡️ '{user_intent}'\n"
        
        for step in path_data:
            idx = self.agent.find_start_node(step['movie'])
            node_id = f"M_{idx}"
            
            if node_id in self.pos_dict:
                x, y = self.pos_dict[node_id]
                path_x.append(x)
                path_y.append(y)
            
            if step['step'] > 0:
                full_text += f"Step {step['step']}: {step['movie']} ({step['confidence']})\n"

        self.path_lines.setData(x=path_x, y=path_y)
        self.label.setText(f"✅ Arrived. Check popup for details.")

        try:
            if hasattr(self, 'narrator'):
                self.label.setText(f"🤖 Generating AI explanation...")
                QApplication.processEvents()
                narrative = self.narrator.explain_journey(start_movie, path_data, user_intent)
                full_text += f"\n\n🤖 AI Insight:\n{narrative}"
            else:
                print("⚠️ Narrator not initialized")
        except Exception as e:
            print(f"⚠️ Narrator failed: {e}")
            full_text += "\n(AI Explanation unavailable)"
        # ----------------------------

        self.label.setText(f"✅ Arrived.")
        
        msg = QMessageBox()
        msg.setWindowTitle("Watch Party Results")
        msg.setText(full_text)
        msg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphWindow()
    window.show()
    sys.exit(app.exec())