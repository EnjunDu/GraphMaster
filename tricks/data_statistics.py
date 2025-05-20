import json
import networkx as nx
import community as community_louvain  # You may need to install this package using: pip install python-louvain

def main():
    # Load the JSON file
    with open('SubChildren.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize counters and sets
    nodes_count = len(data)
    classes = set()
    train_nodes_count = 0
    validation_nodes_count = 0
    test_nodes_count = 0

    # Build an undirected graph
    G = nx.Graph()

    # Iterate over each element in the dataset
    for entry in data:
        node_id = entry['node_id']
        label = entry['label']
        mask = entry['mask']

        # Add label to the classes set
        classes.add(label)

        # Add the node with its attributes to the graph
        G.add_node(node_id, label=label, mask=mask)

        # Count nodes by mask type
        if mask == 'Train':
            train_nodes_count += 1
        elif mask == 'Validation':
            validation_nodes_count += 1
        elif mask == 'Test':
            test_nodes_count += 1

        # Process neighbors and add edges (using set to remove duplicates)
        neighbors = set(entry['neighbors'])
        for neighbor in neighbors:
            # Avoid self-loop if desired (optional)
            if neighbor != node_id:
                G.add_edge(node_id, neighbor)
            # If you want to add self-loops, remove the above condition

    # Compute the number of edges in the graph
    edge_count = G.number_of_edges()
    classes_count = len(classes)

    # Perform Louvain community detection
    partition = community_louvain.best_partition(G)
    communities = set(partition.values())
    community_count = len(communities)

    # Print out the statistics
    print("Nodes count:", nodes_count)
    print("Edges count:", edge_count)
    print("Classes count:", classes_count)
    print("Louvain community count:", community_count)
    print("Train nodes count:", train_nodes_count)
    print("Validation nodes count:", validation_nodes_count)
    print("Test nodes count:", test_nodes_count)

if __name__ == '__main__':
    main()
