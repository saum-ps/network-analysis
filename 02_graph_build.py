# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Day 2: Graph Building & Core Metrics
#
# **Objective**: Build multilayer social graph from CNS data and implement core tie-strength metrics.
#
# **Deliverables**:
# - `build_multilayer_graph(df)` → `nx.MultiDiGraph`
# - `collapse_edges(G)` → weighted `nx.DiGraph`
# - 10 metric callables (≤ 25 lines each)
# - Export `graph.pkl`, `node_metrics.csv`

# %%
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from utils.device import get_device

# %%
# Load the processed data
print("Loading processed CNS data...")
df = pd.read_parquet("parquet/year_month=2013-09.parquet")
print(f"Loaded {len(df)} interactions")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique users: {df['src'].nunique()}")
print(f"Channels: {df['channel'].unique()}")

# %%
def build_multilayer_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build multilayer directed graph from CNS interaction data.
    
    Args:
        df: DataFrame with columns ['src', 'dst', 'timestamp', 'duration', 'channel']
    
    Returns:
        MultiDiGraph with edge attributes: layer, duration, sent
    """
    G = nx.MultiDiGraph()
    
    # Add nodes (all unique users)
    all_users = set(df['src'].unique()) | set(df['dst'].unique())
    G.add_nodes_from(all_users)
    
    # Add edges with layer information
    for _, row in df.iterrows():
        src, dst = row['src'], row['dst']
        duration = row.get('duration', 0)  # SMS has no duration
        channel = row['channel']
        
        # Edge attributes
        edge_attrs = {
            'layer': channel,
            'duration': duration,
            'sent': row['timestamp'],
            'weight': duration + (30 if channel == 'sms' else 0)  # SMS weight = 30
        }
        
        G.add_edge(src, dst, **edge_attrs)
    
    return G

# %%
def collapse_edges(G: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Collapse multilayer graph to weighted directed graph.
    
    Edge weight formula: duration_sec + 30 * sms_count
    
    Args:
        G: MultiDiGraph with edge attributes
    
    Returns:
        DiGraph with weighted edges
    """
    G_collapsed = nx.DiGraph()
    G_collapsed.add_nodes_from(G.nodes())
    
    # Aggregate edges between each pair
    for u, v in G.edges():
        if u == v:  # Skip self-loops
            continue
            
        # Get all edges between u and v
        edges_data = G.get_edge_data(u, v)
        
        total_weight = 0
        total_duration = 0
        sms_count = 0
        call_count = 0
        
        for edge_key, edge_attrs in edges_data.items():
            layer = edge_attrs['layer']
            duration = edge_attrs.get('duration', 0)
            
            if layer == 'sms':
                sms_count += 1
            else:  # call
                call_count += 1
                total_duration += duration
        
        # Calculate weight: duration_sec + 30 * sms_count
        weight = total_duration + 30 * sms_count
        
        if weight > 0:
            G_collapsed.add_edge(u, v, weight=weight, 
                                duration=total_duration, 
                                sms_count=sms_count,
                                call_count=call_count)
    
    return G_collapsed

# %%
# Build the multilayer graph
print("Building multilayer graph...")
G_multilayer = build_multilayer_graph(df)
print(f"Multilayer graph: {G_multilayer.number_of_nodes()} nodes, {G_multilayer.number_of_edges()} edges")

# Collapse to weighted graph
print("Collapsing to weighted graph...")
G = collapse_edges(G_multilayer)
print(f"Weighted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %%
# Core metric functions (≤ 25 lines each)

def top_ties(G: nx.DiGraph, user: int, top_k: int = 5) -> List[Tuple[int, float]]:
    """Get user's top-k ties by weight."""
    if user not in G:
        return []
    
    # Get outgoing edges with weights
    edges = [(v, G[u][v]['weight']) for u, v in G.out_edges(user)]
    edges.extend([(u, G[u][v]['weight']) for u, v in G.in_edges(user)])
    
    # Aggregate weights per neighbor
    neighbor_weights = {}
    for neighbor, weight in edges:
        neighbor_weights[neighbor] = neighbor_weights.get(neighbor, 0) + weight
    
    # Sort by weight and return top-k
    sorted_ties = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_ties[:top_k]

def channel_preference(G: nx.DiGraph, user1: int, user2: int) -> str:
    """Determine preferred channel (sms vs call) for a user pair."""
    if not G.has_edge(user1, user2) and not G.has_edge(user2, user1):
        return "no_interaction"
    
    # Get edge data
    edge_data = G.get_edge_data(user1, user2) or G.get_edge_data(user2, user1) or {}
    
    sms_count = edge_data.get('sms_count', 0)
    call_count = edge_data.get('call_count', 0)
    
    if sms_count > call_count:
        return "sms"
    elif call_count > sms_count:
        return "call"
    else:
        return "equal"

def reciprocity(G: nx.DiGraph, user: int) -> float:
    """Calculate user's reciprocity score (bidirectional connections)."""
    if user not in G:
        return 0.0
    
    outgoing = set(G.successors(user))
    incoming = set(G.predecessors(user))
    
    if not outgoing and not incoming:
        return 0.0
    
    # Count bidirectional connections
    bidirectional = outgoing & incoming
    total_connections = outgoing | incoming
    
    return len(bidirectional) / len(total_connections) if total_connections else 0.0

def detect_circles(G: nx.DiGraph, user: int) -> List[List[int]]:
    """Discover social circles for a user using community detection."""
    if user not in G:
        return []
    
    # Create undirected subgraph around user (2-hop neighborhood)
    neighbors = set(G.predecessors(user)) | set(G.successors(user))
    two_hop = set()
    for neighbor in neighbors:
        two_hop.update(G.predecessors(neighbor))
        two_hop.update(G.successors(neighbor))
    
    # Include user and immediate neighbors
    subgraph_nodes = {user} | neighbors | two_hop
    subgraph = G.subgraph(subgraph_nodes).to_undirected()
    
    # Use Louvain community detection
    try:
        communities = nx.community.louvain_communities(subgraph)
        return [list(comm) for comm in communities if len(comm) > 1]
    except:
        return []

def relationship_trend(G: nx.DiGraph, user1: int, user2: int, df: pd.DataFrame) -> str:
    """Determine if relationship is growing or fading based on temporal data."""
    if not G.has_edge(user1, user2) and not G.has_edge(user2, user1):
        return "no_relationship"
    
    # Get interactions between users
    interactions = df[(df['src'] == user1) & (df['dst'] == user2) | 
                     (df['src'] == user2) & (df['dst'] == user1)]
    
    if len(interactions) < 2:
        return "insufficient_data"
    
    # Split into first and second half
    sorted_interactions = interactions.sort_values('timestamp')
    mid_point = len(sorted_interactions) // 2
    
    first_half = sorted_interactions.iloc[:mid_point]
    second_half = sorted_interactions.iloc[mid_point:]
    
    # Compare activity levels
    first_activity = len(first_half)
    second_activity = len(second_half)
    
    if second_activity > first_activity * 1.2:
        return "growing"
    elif first_activity > second_activity * 1.2:
        return "fading"
    else:
        return "stable"

def avg_reply_delay(G: nx.DiGraph, user1: int, user2: int, df: pd.DataFrame) -> float:
    """Calculate average reply delay between two users."""
    # Get all interactions between users
    interactions = df[(df['src'] == user1) & (df['dst'] == user2) | 
                     (df['src'] == user2) & (df['dst'] == user1)].sort_values('timestamp')
    
    if len(interactions) < 2:
        return float('inf')
    
    delays = []
    for i in range(len(interactions) - 1):
        current = interactions.iloc[i]
        next_interaction = interactions.iloc[i + 1]
        
        # If next interaction is from different user, it's a reply
        if current['src'] != next_interaction['src']:
            delay = next_interaction['timestamp'] - current['timestamp']  # Integer difference
            delays.append(delay)
    
    return np.mean(delays) if delays else float('inf')

def extrovert_score(G: nx.DiGraph, user: int) -> float:
    """Calculate extrovert score based on outgoing connections and bridge role."""
    if user not in G:
        return 0.0
    
    # Outgoing degree (initiative)
    out_degree = G.out_degree(user)
    
    # Bridge score (connectivity between communities)
    neighbors = list(G.successors(user)) + list(G.predecessors(user))
    if len(neighbors) < 2:
        return out_degree
    
    # Calculate local clustering coefficient
    local_clustering = nx.clustering(G, user) if len(neighbors) > 1 else 0
    
    # Extrovert score: high outgoing degree, low clustering (bridge role)
    extrovert_score = out_degree * (1 - local_clustering)
    
    return extrovert_score

def churn_drop(G: nx.DiGraph, user: int, df: pd.DataFrame, weeks: int = 4) -> float:
    """Detect weekly inbound-drop churn for a user."""
    if user not in G:
        return 0.0
    
    # Get all interactions involving user
    user_interactions = df[(df['src'] == user) | (df['dst'] == user)].sort_values('timestamp')
    
    if len(user_interactions) < 10:
        return 0.0
    
    # Split into weekly periods (timestamps are relative integers)
    min_time = user_interactions['timestamp'].min()
    max_time = user_interactions['timestamp'].max()
    total_time = max_time - min_time
    
    if total_time < weeks * 7:  # Need at least 4 weeks of data
        return 0.0
    
    # Calculate weekly inbound counts
    weekly_inbound = []
    week_duration = total_time // weeks
    
    for week in range(weeks):
        week_start = min_time + (week * week_duration)
        week_end = week_start + week_duration if week < weeks - 1 else max_time + 1
        
        week_interactions = user_interactions[
            (user_interactions['timestamp'] >= week_start) & 
            (user_interactions['timestamp'] < week_end) &
            (user_interactions['dst'] == user)
        ]
        
        weekly_inbound.append(len(week_interactions))
    
    # Calculate drop rate
    if weekly_inbound[0] == 0:
        return 0.0
    
    drop_rate = (weekly_inbound[0] - weekly_inbound[-1]) / weekly_inbound[0]
    return max(0, drop_rate)

def find_spam_nodes(G: nx.DiGraph, threshold: float = 0.8) -> List[int]:
    """Find potential spam/harassment nodes based on activity patterns."""
    spam_nodes = []
    
    for node in G.nodes():
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        
        if out_degree == 0:
            continue
        
        # Calculate spam indicators
        reply_ratio = in_degree / out_degree if out_degree > 0 else 0
        activity_ratio = out_degree / (out_degree + in_degree) if (out_degree + in_degree) > 0 else 0
        
        # High outgoing, low incoming, low replies = potential spam
        if activity_ratio > threshold and reply_ratio < (1 - threshold):
            spam_nodes.append(node)
    
    return spam_nodes

# %%
# Test the core metrics
print("Testing core metrics...")

# Test top_ties
test_user = list(G.nodes())[0]
top_connections = top_ties(G, test_user)
print(f"Top ties for user {test_user}: {top_connections[:3]}")

# Test channel_preference
if G.number_of_edges() > 0:
    edge = list(G.edges())[0]
    preference = channel_preference(G, edge[0], edge[1])
    print(f"Channel preference for {edge}: {preference}")

# Test reciprocity
reciprocity_score = reciprocity(G, test_user)
print(f"Reciprocity for user {test_user}: {reciprocity_score:.3f}")

# Test new metrics
print("\nTesting advanced metrics...")

# Test detect_circles
circles = detect_circles(G, test_user)
print(f"Social circles for user {test_user}: {len(circles)} circles found")

# Test extrovert_score
extrovert = extrovert_score(G, test_user)
print(f"Extrovert score for user {test_user}: {extrovert:.2f}")

# Test spam detection
spam_nodes = find_spam_nodes(G)
print(f"Potential spam nodes found: {len(spam_nodes)}")

# Test relationship trend (if edge exists)
if G.number_of_edges() > 0:
    edge = list(G.edges())[0]
    trend = relationship_trend(G, edge[0], edge[1], df)
    print(f"Relationship trend for {edge}: {trend}")

# %%
# Calculate comprehensive metrics for all users
print("Calculating comprehensive metrics for all users...")

user_metrics = []
for user in G.nodes():
    top_5_ties = top_ties(G, user, 5)
    reciprocity_score = reciprocity(G, user)
    extrovert_score_val = extrovert_score(G, user)
    circles = detect_circles(G, user)
    churn_rate = churn_drop(G, user, df)
    
    # Get user's total activity
    out_degree = G.out_degree(user)
    in_degree = G.in_degree(user)
    total_activity = out_degree + in_degree
    
    user_metrics.append({
        'user_id': user,
        'top_ties': top_5_ties,
        'reciprocity': reciprocity_score,
        'extrovert_score': extrovert_score_val,
        'num_circles': len(circles),
        'churn_rate': churn_rate,
        'out_degree': out_degree,
        'in_degree': in_degree,
        'total_activity': total_activity
    })

# %%
# Export results
print("Exporting results...")

# Save graph
with open('graph.pkl', 'wb') as f:
    pickle.dump(G, f)
print("Saved graph.pkl")

# Save metrics
metrics_df = pd.DataFrame(user_metrics)
metrics_df.to_csv('node_metrics.csv', index=False)
print("Saved node_metrics.csv")

print(f"Day 2 complete! Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges") 
