#!/usr/bin/env python3
"""
Test script for network analysis - run this locally to verify the logic works
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io

def test_network_analysis():
    """Test the network analysis logic locally"""
    
    print("=== Testing Network Analysis Locally ===")
    
    # Load the test data
    try:
        df = pd.read_csv('edges.csv')
        print(f"✓ Successfully loaded edges.csv")
        print(f"  Data shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First few rows:\n{df.head()}")
    except Exception as e:
        print(f"✗ Failed to load edges.csv: {e}")
        return
    
    # Check if we have the required columns
    if 'source' not in df.columns or 'target' not in df.columns:
        print("✗ Missing required columns 'source' and 'target'")
        return
    
    # Create the network graph
    try:
        print("\n=== Creating Network Graph ===")
        G = nx.Graph()
        
        # Add edges
        for _, row in df.iterrows():
            G.add_edge(row['source'], row['target'])
        
        print(f"✓ Graph created successfully")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Node list: {list(G.nodes())}")
        print(f"  Edge list: {list(G.edges())}")
        
    except Exception as e:
        print(f"✗ Failed to create graph: {e}")
        return
    
    # Calculate network metrics
    try:
        print("\n=== Calculating Network Metrics ===")
        
        # Answer 1: Edge count
        edge_count = G.number_of_edges()
        print(f"✓ Edge count: {edge_count}")
        
        # Answer 2: Highest degree node
        degrees = dict(G.degree())
        highest_degree_node = max(degrees, key=degrees.get)
        print(f"✓ Highest degree node: {highest_degree_node} (degree: {degrees[highest_degree_node]})")
        
        # Answer 3: Average degree
        avg_degree = sum(degrees.values()) / len(degrees)
        print(f"✓ Average degree: {avg_degree:.2f}")
        
        # Answer 4: Network density
        density = nx.density(G)
        print(f"✓ Network density: {density:.4f}")
        
        # Answer 5: Shortest path between Alice and Eve
        try:
            shortest_path = nx.shortest_path_length(G, 'Alice', 'Eve')
            print(f"✓ Shortest path between Alice and Eve: {shortest_path}")
        except nx.NetworkXNoPath:
            print("✗ No path exists between Alice and Eve")
            shortest_path = "No path exists"
        except Exception as e:
            print(f"✗ Error calculating shortest path: {e}")
            shortest_path = "Error"
        
    except Exception as e:
        print(f"✗ Failed to calculate metrics: {e}")
        return
    
    # Generate network visualization
    try:
        print("\n=== Generating Network Visualization ===")
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=1000, font_size=10, font_weight='bold')
        plt.title("Network Graph")
        
        # Save to buffer and convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        print(f"✓ Network graph generated successfully")
        print(f"  Base64 length: {len(img_base64)} characters")
        print(f"  Image size: {len(img_base64) * 3 // 4} bytes (approx)")
        
    except Exception as e:
        print(f"✗ Failed to generate network visualization: {e}")
        img_base64 = f"Error: {str(e)}"
    
    # Generate degree histogram
    try:
        print("\n=== Generating Degree Histogram ===")
        plt.figure(figsize=(8, 6))
        degree_values = list(degrees.values())
        plt.hist(degree_values, bins=range(min(degree_values), max(degree_values) + 2), 
                color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save to buffer and convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        histogram_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        print(f"✓ Degree histogram generated successfully")
        print(f"  Base64 length: {len(histogram_base64)} characters")
        print(f"  Image size: {len(histogram_base64) * 3 // 4} bytes (approx)")
        
    except Exception as e:
        print(f"✗ Failed to generate degree histogram: {e}")
        histogram_base64 = f"Error: {str(e)}"
    
    # Create final results
    print("\n=== Final Results ===")
    final_results = {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": round(avg_degree, 2),
        "density": round(density, 4),
        "shortest_path_alice_eve": shortest_path,
        "network_graph": img_base64,
        "degree_histogram": histogram_base64
    }
    
    # Save results to file
    try:
        with open('test_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        print("✓ Results saved to test_results.json")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"✓ All tests completed successfully!")
    print(f"✓ Network has {edge_count} edges")
    print(f"✓ {highest_degree_node} has the highest degree")
    print(f"✓ Average degree: {avg_degree:.2f}")
    print(f"✓ Network density: {density:.4f}")
    print(f"✓ Shortest path Alice→Eve: {shortest_path}")
    print(f"✓ Generated network visualization ({len(img_base64)} chars)")
    print(f"✓ Generated degree histogram ({len(histogram_base64)} chars)")
    
    return final_results

if __name__ == "__main__":
    test_network_analysis()

