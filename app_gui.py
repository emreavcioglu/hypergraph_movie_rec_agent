import sys
import json
import os
import networkx as nx
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# ----------- CONFIG
DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "movies.csv")
HYPERGRAPH_FILE = os.path.join(DATA_DIR, "hypergraph.json")
MOVIES_PER_CLUSTER = 5


class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movie Hypergraph Visualization")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        self.label = QLabel("Loading hypergraph...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; color: #00d4ff;")
        layout.addWidget(self.label)

        # Graph widget
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addPlot()
        self.view.setAspectLocked()
        self.view.hideAxis('bottom')
        self.view.hideAxis('left')
        layout.addWidget(self.graph_widget)

        # Data storage
        self.scatter = pg.ScatterPlotItem(size = 10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.view.addItem(self.scatter)
        self.text_items = []
        self.build_graph_data()


    def build_graph_data(self):
        if not os.path.exists(METADATA_FILE) or not os.path.exists(HYPERGRAPH_FILE):
            self.label.setText("Required data files are missing.")
            return
        
        print("Loading data")

        with open(HYPERGRAPH_FILE, 'r') as f:
            hypergraph = json.load(f)

        df = pd.read_csv(METADATA_FILE)

        self.G = nx.Graph()

        self.node_colors = []
        self.node_sizes = []
        self.node_labels = []
        self.pos =  {}

        print("Building connections")

        for theme_name, data in hypergraph.items():
            self.G.add_node(theme_name, type='theme')

            indices = data['indices'][:MOVIES_PER_CLUSTER]


            for idx in indices:
                movie_id = f"M_{idx}"
                movie_title = str(df.iloc[idx]['title'])
                self.G.add_node(movie_id, type='movie', title=movie_title)
                self.G.add_edge(theme_name, movie_id)


        print("Calculating layout")
        self.label.setText(f"Calculating physics for {len(self.G.nodes)} nodes")

        pos_dict = nx.spring_layout(self.G, dim=2, k=0.5, iterations=50, seed=42)

        spots = []
        for node, (x,y) in pos_dict.items():
            node_type = self.G.nodes[node].get('type', 'movie')
            if node_type == 'theme':
                color = (255,50,50,200)
                size = 20
                symbol = 'o'

                short_name = node.replace("Theme: ", "").replace("Genre: ", "")
                text = pg.TextItem(short_name, anchor = (0.5,0.5), color=(255,255,255))
                text.setPos(x,y)
                self.view.addItem(text)
            else:
                color = (50,150,255,150)
                size = 8
                symbol = 'o'
            
            spots.append({'pos': (x,y), 'brush': pg.mkBrush(color), 'size': size, 'symbol': symbol, 'data': node})

        self.scatter.addPoints(spots)

        print("Drawing lines")

        for u,v in self.G.edges():
            if u in pos_dict and v in pos_dict:
                x1,y1 = pos_dict[u]
                x2,y2 = pos_dict[v]
                line = pg.PlotCurveItem(x=[x1,x2], y=[y1,y2], pen=pg.mkPen(100,100,100,50))
                self.view.addItem(line)

        self.view.addItem(self.scatter)

        self.label.setText(f"Hypergraph Ready: {len(self.G.nodes)} Nodes")
        print("GUI ready")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphWindow()
    window.show()
    sys.exit(app.exec())