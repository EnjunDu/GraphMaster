# data_sample.py

import networkx as nx
import json
import random
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt


def load_graph_from_edges_file(edges_file: str) -> nx.Graph:
    """
    Read edges from the given edge list file and construct an undirected graph.
    :param edges_file: edge file path, two integers per line, representing (node1, node2)
    :return: constructed NetworkX undirected graph
    """
    edges = []
    with open(edges_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                node1, node2 = map(int, line.split())
                edges.append((node1, node2))
            except ValueError:
                print(f"[Warning] Invalid line in edges file: '{line}'")

    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def run_louvain_community_detection(G: nx.Graph):
    """
    Perform Louvain community detection on graph G and return the community partition dictionary and the total number of communities.
    :param G: input NetworkX graph
    :return:
        partition: dict, key is node, value is community number
        num_communities: int, total number of communities
    """
    partition = community_louvain.best_partition(G)
    num_communities = max(partition.values()) + 1 if partition else 0
    return partition, num_communities


def sort_communities_by_size(partition: dict) -> list:
    """
    Returns a sorted list of community numbers from largest to smallest based on community size.
    :param partition: Louvain community partition result
    :return: list, community numbers sorted in descending order by size
    """
    # Collect community -> Node list
    community_nodes = {}
    for node, comm_id in partition.items():
        community_nodes.setdefault(comm_id, []).append(node)

    # Sort by community size
    sorted_communities = sorted(
        community_nodes.keys(),
        key=lambda c: len(community_nodes[c]),
        reverse=True
    )
    return sorted_communities


def visualize_communities(G: nx.Graph, partition: dict, sampled_nodes=None, title="Louvain Community Detection"):
    """
    Visualize network graph and community partition results, support highlighting sampled nodes.
    :param G: NetworkX graph
    :param partition: Louvain community partition result
    :param sampled_nodes: list of nodes to be highlighted
    :param title: image title
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Calculate layout (random, can also be changed to other)

    # Use community number as color
    colors = [partition[n] for n in G.nodes()]
    nx.draw(
        G, pos,
        node_color=colors,
        node_size=30,
        with_labels=False,
        edge_color="gray",
        alpha=0.3
    )

    # If there is a sampling node, it will be highlighted in the graph
    if sampled_nodes:
        highlight_nodes = set(sampled_nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(highlight_nodes),
            node_color="red",
            node_size=50
        )

    plt.title(title)
    plt.show()


if __name__ == "__main__":
    edges_file = "edges.txt"  # Default edge file path
    print(f"[data_sample.py] Loading edges from {edges_file} ...")
    G = load_graph_from_edges_file(edges_file)
    print(f"[data_sample.py] Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("[data_sample.py] Running Louvain community detection...")
    partition, num_communities = run_louvain_community_detection(G)
    print(f"[data_sample.py] Detected {num_communities} communities.")

    # Sort by community size
    sorted_communities = sort_communities_by_size(partition)

    # Print sorting results
    print("Communities sorted by size (descending):")
    for idx, comm in enumerate(sorted_communities):
        # Count the number of nodes in the community
        node_count = sum(1 for node, c in partition.items() if c == comm)
        print(f"Rank {idx}: Community {comm} has {node_count} nodes")

    # Read user input
    print("\nEnter 'start_community_index end_community_index sample_size_k', e.g. '0 1 50'")
    user_input = input(">>> ")
    try:
        start_community_index, end_community_index, sample_size_k = map(int, user_input.split())
    except:
        print("[Error] Invalid input. Please input 3 integers separated by space.")
        exit(1)

    # defensive check
    if start_community_index < 0 or start_community_index >= len(sorted_communities):
        print("[Warning] start_community_index out of range. Force set to 0.")
        start_community_index = 0
    if end_community_index < start_community_index:
        print("[Warning] end_community_index < start_community_index. Force set to the same as start.")
        end_community_index = start_community_index
    if end_community_index >= len(sorted_communities):
        print(f"[Warning] end_community_index out of range. Force set to {len(sorted_communities) - 1}.")
        end_community_index = len(sorted_communities) - 1

    # Collect multiple community nodes
    selected_nodes_all = []
    for idx in range(start_community_index, end_community_index + 1):
        comm_id = sorted_communities[idx]
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        selected_nodes_all.extend(comm_nodes)

    total_nodes_count = len(selected_nodes_all)
    print(f"[data_sample.py] Selected communities from rank {start_community_index} to {end_community_index}.")
    print(f"[data_sample.py] Total nodes in these communities: {total_nodes_count}")

    # sampling
    sample_k = min(sample_size_k, total_nodes_count)
    sampled_nodes = random.sample(selected_nodes_all, sample_k)
    sampled_nodes_sorted = sorted(sampled_nodes)
    print(f"[data_sample.py] Randomly sampled {len(sampled_nodes_sorted)} nodes. Below are the sorted IDs:\n")
    output_str = ", ".join(map(str, sampled_nodes_sorted))
    print(output_str)

    # write file
    output_path = "selected_community_sampled_nodes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    print(f"[data_sample.py] Sampled nodes saved to {output_path}")

    # (Optional) Visualize the results
    visualize_communities(
        G, partition,
        sampled_nodes=sampled_nodes_sorted,
        title=f"Louvain - Sample {sample_k} from communities [{start_community_index}..{end_community_index}]"
    )
