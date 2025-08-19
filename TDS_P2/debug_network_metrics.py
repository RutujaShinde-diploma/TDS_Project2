#!/usr/bin/env python3

import pandas as pd
import networkx as nx
from pathlib import Path

def test_basic_network_metrics():
    """Test the basic network metrics extraction directly"""
    
    # Check if edges.csv exists
    edges_file = Path("edges.csv")
    print(f"Looking for edges.csv: {edges_file.exists()}")
    
    if not edges_file.exists():
        print("❌ edges.csv not found")
        return {}
    
    print("✅ edges.csv found")
    
    # Load edges data
    df = pd.read_csv(edges_file)
    print(f"Loaded CSV with columns: {list(df.columns)}")
    print(f"CSV shape: {df.shape}")
    print(f"First few rows:")
    print(df.head())
    
    if 'source' not in df.columns or 'target' not in df.columns:
        print(f"❌ Missing required columns. Found: {list(df.columns)}")
        return {}
    
    # Create simple graph for basic metrics
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate basic metrics
    edge_count = G.number_of_edges()
    node_count = G.number_of_nodes()
    
    # Calculate degrees
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get) if degrees else "Unknown"
    average_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Calculate density
    density = nx.density(G) if node_count > 1 else 0
    
    # Try shortest path
    shortest_path = 0
    try:
        if 'Alice' in G.nodes and 'Eve' in G.nodes:
            shortest_path = nx.shortest_path_length(G, 'Alice', 'Eve')
            print(f"✅ Calculated shortest path: {shortest_path}")
        else:
            shortest_path = 0
            print(f"❌ Alice or Eve not found. Available nodes: {list(G.nodes())}")
    except Exception as path_error:
        shortest_path = 0
        print(f"❌ Path calculation error: {path_error}")
    
    result = {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": round(average_degree, 2),
        "density": round(density, 4),
        "shortest_path_alice_eve": shortest_path,
        "extraction_method": "csv_fallback"
    }
    
    print(f"✅ Final metrics: {result}")
    return result

if __name__ == "__main__":
    test_basic_network_metrics()
