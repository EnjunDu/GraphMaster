# data_sample.py
import networkx as nx
import json
import random
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

# =========================
# 1. Read edge data and build a graph
# =========================
edges_file = "edges.txt"  # Edge file path
edges = []

# Read edge files and parse node pairs
with open(edges_file, "r") as file:
    for line in file:
        node1, node2 = map(int, line.strip().split())  # Parse the two nodes in each row
        edges.append((node1, node2))

# Building a NetworkX Graph
G = nx.Graph()
G.add_edges_from(edges)

# =========================
# 2. Running Louvain community detection
# =========================
partition = community_louvain.best_partition(G)  # Computing Louvain Community
num_communities = max(partition.values()) + 1  # Count the number of communities
print(f"Detected {num_communities} communities")

# Count the nodes in each community
community_nodes = {}
for node, comm in partition.items():
    community_nodes.setdefault(comm, []).append(node)

# =========================
# 3. Communities are sorted by the number of nodes from largest to smallest
# =========================
sorted_communities = sorted(
    community_nodes.keys(),
    key=lambda c: len(community_nodes[c]),
    reverse=True
)

print("Communities sorted by size (descending):")
for idx, comm in enumerate(sorted_communities):
    print(f"Rank {idx}: Community {comm} has {len(community_nodes[comm])} nodes")

# =========================
# 4. Read user input
# =========================
user_input = input("\nEnter start_community_index, end_community_index, sample_size_k (separated by space): ")
start_community_index, end_community_index, sample_size_k = map(int, user_input.split())

# Safety check: if end_community_index exceeds the available range, truncate to the end
end_community_index = min(end_community_index, len(sorted_communities) - 1)

# =========================
# 5. Merge nodes of interval communities
# =========================
selected_nodes_all = []
for idx in range(start_community_index, end_community_index + 1):
    comm_id = sorted_communities[idx]
    selected_nodes_all.extend(community_nodes[comm_id])

total_nodes_count = len(selected_nodes_all)
print(f"Selected communities from rank {start_community_index} to {end_community_index}.")
print(f"Total nodes in these communities: {total_nodes_count}")

# =========================
# 6. Sample a specified number k of nodes
# =========================
sample_k = min(sample_size_k, total_nodes_count)
sampled_nodes = random.sample(selected_nodes_all, sample_k)
print(f"Randomly sampled {len(sampled_nodes)} nodes from the selected communities.")

# =========================
# 7. Output sampling results (after sorting)
# =========================
output_path = "selected_community_sampled_nodes.json"
sampled_nodes_sorted = sorted(sampled_nodes)  # Sort the sampling nodes from small to large
output_str = ", ".join(map(str, sampled_nodes_sorted))
print(output_str)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_str)

print(f"Sampled nodes saved to {output_path}")

# =========================
# 8. Visualization (optional)
# =========================
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)  # Calculate layout
# Different communities use different colors
colors = [partition[n] for n in G.nodes()]
nx.draw(G, pos, node_color=colors, node_size=30, with_labels=False, edge_color="gray", alpha=0.3)

# Highlight sampling node
highlight_nodes = set(sampled_nodes_sorted)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(highlight_nodes),
    node_color="red",
    node_size=50
)

plt.title(f"Louvain Community Detection - Sampled {sample_k} nodes from communities rank {start_community_index}~{end_community_index}")
plt.show()
