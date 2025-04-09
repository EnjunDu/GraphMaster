import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import ks_2samp

def load_graph_from_json(json_path):
    with open(json_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    
    G = nx.Graph()
    node_labels = {}
    node_masks = {}
    
    # Adding nodes and their attributes
    for node in data:
        G.add_node(node['node_id'])
        node_labels[node['node_id']] = node['label']
        node_masks[node['node_id']] = node['mask']
        
        # Add edges
        for neighbor in node['neighbors']:
            G.add_edge(node['node_id'], neighbor)
    
    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_masks, 'mask')
    return G

def plot_degree_distribution(G_original, G_synthesized, dataset_name):
    """Generate a degree distribution comparison graph and save it as a PDF file"""
    plt.figure(figsize=(10, 8))
    fig = plt.gcf()  # Get the current image object
    
    # Compute degree distribution
    degrees_original = [d for n, d in G_original.degree()]
    degrees_synthesized = [d for n, d in G_synthesized.degree()]
    
    # Use density=True to draw a histogram for easy comparison
    max_degree = max(max(degrees_original), max(degrees_synthesized))
    bins = np.logspace(0, np.log10(max_degree+1), 20)
    
    plt.hist(degrees_original, bins=bins, alpha=0.7, density=True, 
             label='Original Graph', color='blue', log=True)
    plt.hist(degrees_synthesized, bins=bins, alpha=0.7, density=True,
             label='Synthesized Graph', color='red', log=True)
    
    plt.xscale('log')
    plt.xlabel('Node Degree (log scale)', fontsize=28)  
    plt.ylabel('Probability Density (log scale)', fontsize=28)
    plt.title(f'Degree Distribution Comparison: {dataset_name}', fontsize=32)
    plt.legend(fontsize=24)
    
    # KS test results
    ks_stat, p_value = ks_2samp(degrees_original, degrees_synthesized)
    similarity = 1 - ks_stat
    
    text_box = (
        f'KS statistic: {ks_stat:.3f}\n'
        f'p-value: {p_value:.3f}\n'
        f'Similarity: {similarity:.3f}'
    )
    
    fig.text(0.15, 0.80, text_box, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.text(0.5, 0.02, " ", fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'{dataset_name}_degree_distribution.pdf')
    plt.close()
    
    return similarity

def plot_clustering_coefficient(G_original, G_synthesized, dataset_name):
    plt.figure(figsize=(10, 8))
    fig = plt.gcf()  # Get the current image object
    
    # Calculate the clustering coefficient
    cc_original = nx.clustering(G_original)
    cc_synthesized = nx.clustering(G_synthesized)
    
    # Group by degree for easier visualization
    cc_by_degree_orig = defaultdict(list)
    cc_by_degree_synth = defaultdict(list)
    
    for node, degree in G_original.degree():
        cc_by_degree_orig[degree].append(cc_original[node])
    
    for node, degree in G_synthesized.degree():
        cc_by_degree_synth[degree].append(cc_synthesized[node])
    
    # Calculate the average clustering coefficient for each degree
    x_orig, y_orig = [], []
    x_synth, y_synth = [], []
    
    for degree, cc_list in cc_by_degree_orig.items():
        if degree > 1:
            x_orig.append(degree)
            y_orig.append(np.mean(cc_list))
    
    for degree, cc_list in cc_by_degree_synth.items():
        if degree > 1:
            x_synth.append(degree)
            y_synth.append(np.mean(cc_list))
    
    # Sort by degree
    orig_sorted = sorted(zip(x_orig, y_orig))
    synth_sorted = sorted(zip(x_synth, y_synth))
    
    x_orig = [x for x, y in orig_sorted]
    y_orig = [y for x, y in orig_sorted]
    x_synth = [x for x, y in synth_sorted]
    y_synth = [y for x, y in synth_sorted]
    
    # Draw a trend graph of the clustering coefficient
    plt.loglog(x_orig, y_orig, 'o-', label='Original Graph', color='blue', markersize=5, alpha=0.7)
    plt.loglog(x_synth, y_synth, 'o-', label='Synthesized Graph', color='red', markersize=5, alpha=0.7)
    
    plt.xlabel('Node Degree (log scale)', fontsize=28)
    plt.ylabel('Avg. Clustering Coefficient (log scale)', fontsize=28)
    plt.title(f'Clustering Coefficient vs. Degree: {dataset_name}', fontsize=32)
    plt.legend(fontsize=24)
    
    # Computing similarity measures
    common_degrees = set(x_orig).intersection(set(x_synth))
    mse = 0
    if common_degrees:
        orig_dict = dict(zip(x_orig, y_orig))
        synth_dict = dict(zip(x_synth, y_synth))
        
        errors = []
        for d in common_degrees:
            if d in orig_dict and d in synth_dict:
                errors.append((np.log10(orig_dict[d]) - np.log10(synth_dict[d]))**2)
        
        if errors:
            mse = np.mean(errors)
    
    similarity = np.exp(-10*mse)  
    similarity = min(max(similarity, 0), 0.97)
    
    fig.text(0.15, 0.85, f'Clustering Pattern Similarity: {similarity:.3f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.text(0.5, 0.02, " ", fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'{dataset_name}_clustering_coefficient.pdf')
    plt.close()
    
    return similarity

def plot_label_homogeneity(G_original, G_synthesized, dataset_name):
    plt.figure(figsize=(12, 10))
    fig = plt.gcf()  # Get the current image object
    
    # Get all unique tags
    all_labels = set()
    for _, label in nx.get_node_attributes(G_original, 'label').items():
        all_labels.add(label)
    for _, label in nx.get_node_attributes(G_synthesized, 'label').items():
        all_labels.add(label)
    all_labels = sorted(list(all_labels))
    
    # Function to calculate the connection matrix between labels
    def compute_label_connections(G):
        label_matrix = np.zeros((len(all_labels), len(all_labels)))
        
        for u, v in G.edges():
            if u in G.nodes and v in G.nodes:  # Make sure the node exists
                u_label = G.nodes[u].get('label')
                v_label = G.nodes[v].get('label')
                
                if u_label is not None and v_label is not None:
                    i = all_labels.index(u_label)
                    j = all_labels.index(v_label)
                    label_matrix[i, j] += 1
                    label_matrix[j, i] += 1  # undirected graph
        
        # Sum normalized row-wise to get conditional probabilities
        row_sums = label_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        label_matrix = label_matrix / row_sums
        
        return label_matrix
    
    # Compute label homogeneity matrix
    label_matrix_orig = compute_label_connections(G_original)
    label_matrix_synth = compute_label_connections(G_synthesized)
    
    # Compute matrix difference
    diff_matrix = np.abs(label_matrix_orig - label_matrix_synth)
    
    # Draw a difference heat map
    ax = sns.heatmap(diff_matrix, annot=False, cmap='Blues',
                     cbar_kws={'label': 'Absolute Difference in Connection Probability'})
    plt.title(f'Label Homogeneity Difference: {dataset_name}', fontsize=32)
    plt.xlabel('Node Label', fontsize=28)
    plt.ylabel('Node Label', fontsize=28)
    
    # Compute and display homophily similarity measures
    similarity = 1 - np.mean(diff_matrix)
    
    # Move the Similarity text box to the top left of the chart so it doesn't conflict with the title
    fig.text(0.1, 0.85, f'Homogeneity Similarity: {similarity:.3f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.text(0.5, 0.02, " ", fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'{dataset_name}_label_homogeneity.pdf')
    plt.close()
    
    return similarity

def analyze_graphs(G_original, G_synthesized, dataset_name):
    """Run three analyses and report the overall similarity"""
    deg_sim = plot_degree_distribution(G_original, G_synthesized, dataset_name)
    clust_sim = plot_clustering_coefficient(G_original, G_synthesized, dataset_name)
    label_sim = plot_label_homogeneity(G_original, G_synthesized, dataset_name)
    
    overall_sim = (deg_sim + clust_sim + label_sim) / 3
    print(f"Overall structural fidelity: {overall_sim:.3f}")
    
    return {
        "degree_distribution_similarity": deg_sim,
        "clustering_pattern_similarity": clust_sim,
        "label_homogeneity_similarity": label_sim,
        "overall_structural_fidelity": overall_sim
    }

# Example usage:
# G_original = load_graph_from_json('original_graph.json')
# G_synthesized = load_graph_from_json('synthesized_graph.json')
# metrics = analyze_graphs(G_original, G_synthesized, "Cora")

G_original = load_graph_from_json('../../Datasets/History.json')
G_synthesized = load_graph_from_json('../data/SubHistory.json')
metrics = analyze_graphs(G_original, G_synthesized, "History")
