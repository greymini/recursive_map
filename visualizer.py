import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
import os

from config import ENTITY_TYPES


# Color mapping for entity types
ENTITY_COLORS = {
    "Person": "#FF6B6B",
    "Organization": "#4ECDC4",
    "Product": "#45B7D1",
    "Location": "#96CEB4",
    "Event": "#FFEAA7",
    "Technology": "#DDA0DD",
    "Unknown": "#95A5A6"
}


def create_networkx_graph(knowledge_graph: Dict) -> nx.DiGraph:
    """Create NetworkX directed graph from knowledge graph JSON."""
    G = nx.DiGraph()
    
    # Add nodes
    for node in knowledge_graph.get("nodes", []):
        G.add_node(
            node["id"],
            label=node.get("label", ""),
            type=node.get("type", "Unknown"),
            confidence=node.get("confidence", 0.0),
            context=node.get("context", "")
        )
    
    # Add edges
    for edge in knowledge_graph.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge.get("relationship", ""),
            confidence=edge.get("confidence", 0.0),
            description=edge.get("description", "")
        )
    
    return G


def visualize_graph(knowledge_graph: Dict, output_path: Optional[str] = None,
                   figsize: tuple = (16, 12), node_size_multiplier: int = 500,
                   edge_width_multiplier: float = 2.0) -> None:
    """Visualize knowledge graph using NetworkX and matplotlib."""
    G = create_networkx_graph(knowledge_graph)
    
    if len(G.nodes()) == 0:
        print("No nodes to visualize")
        return
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Calculate node sizes based on degree (number of relationships)
    node_sizes = []
    for node_id in G.nodes():
        degree = G.degree(node_id)
        node_sizes.append(max(300, degree * node_size_multiplier))
    
    # Get node colors based on entity type
    node_colors = []
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get("type", "Unknown")
        node_colors.append(ENTITY_COLORS.get(node_type, ENTITY_COLORS["Unknown"]))
    
    # Calculate edge widths based on confidence
    edge_widths = []
    for u, v in G.edges():
        confidence = G[u][v].get("confidence", 0.5)
        edge_widths.append(max(0.5, confidence * edge_width_multiplier))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        linewidths=2,
        edgecolors='black'
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw edge labels (relationship types)
    edge_labels = {}
    for u, v in G.edges():
        relationship = G[u][v].get("relationship", "")
        if relationship:
            edge_labels[(u, v)] = relationship
    
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=7,
            font_color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # Draw node labels
    labels = {node_id: G.nodes[node_id].get("label", node_id) for node_id in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=8,
        font_weight='bold'
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=color, label=entity_type)
        for entity_type, color in ENTITY_COLORS.items()
        if entity_type in [G.nodes[n].get("type", "Unknown") for n in G.nodes()]
    ]
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def export_graph_svg(knowledge_graph: Dict, output_path: str, 
                     figsize: tuple = (16, 12)) -> None:
    """Export graph visualization as SVG."""
    G = create_networkx_graph(knowledge_graph)
    
    if len(G.nodes()) == 0:
        print("No nodes to visualize")
        return
    
    plt.figure(figsize=figsize)
    
    node_sizes = [max(300, G.degree(node_id) * 500) for node_id in G.nodes()]
    node_colors = [ENTITY_COLORS.get(G.nodes[node_id].get("type", "Unknown"), 
                                     ENTITY_COLORS["Unknown"]) for node_id in G.nodes()]
    edge_widths = [max(0.5, G[u][v].get("confidence", 0.5) * 2.0) for u, v in G.edges()]
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.8, linewidths=2, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray',
                          arrows=True, arrowsize=20, arrowstyle='->')
    
    # Draw edge labels (relationship types)
    edge_labels = {}
    for u, v in G.edges():
        relationship = G[u][v].get("relationship", "")
        if relationship:
            edge_labels[(u, v)] = relationship
    
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=7,
            font_color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    labels = {node_id: G.nodes[node_id].get("label", node_id) for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")
    plt.close()

