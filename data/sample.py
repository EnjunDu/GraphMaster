import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import ks_2samp
import matplotlib.colorbar as colorbar

def load_graph_from_json(json_path):
    with open(json_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    
    G = nx.Graph()
    node_labels = {}
    node_masks = {}
    
    # 添加节点及其属性
    for node in data:
        G.add_node(node['node_id'])
        node_labels[node['node_id']] = node['label']
        node_masks[node['node_id']] = node['mask']
        
        # 添加边
        for neighbor in node['neighbors']:
            G.add_edge(node['node_id'], neighbor)
    
    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_masks, 'mask')
    return G

def plot_degree_distribution(G_original, G_synthesized, dataset_name):
    """Generate a beautiful scatter plot visualization with pure white background"""
    plt.figure(figsize=(12, 10))
    fig = plt.gcf()
    
    # Set pure white background
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Remove grid for cleaner look
    plt.grid(False)
    
    # Calculate degree distributions
    degrees_original = [d for n, d in G_original.degree()]
    degrees_synthesized = [d for n, d in G_synthesized.degree()]
    
    # Calculate probability mass functions for each discrete degree
    def calculate_pmf(degrees):
        counts = {}
        for d in degrees:
            counts[d] = counts.get(d, 0) + 1
        
        total = len(degrees)
        pmf = {d: count/total for d, count in counts.items()}
        
        # Sort by degree for plotting
        sorted_items = sorted(pmf.items())
        x = [item[0] for item in sorted_items]
        y = [item[1] for item in sorted_items]
        
        return x, y, counts
    
    # Calculate PMFs
    x_orig, y_orig, counts_orig = calculate_pmf(degrees_original)
    x_synth, y_synth, counts_synth = calculate_pmf(degrees_synthesized)
    
    # Find common degree values
    common_degrees = set(x_orig).intersection(set(x_synth))
    orig_dict = dict(zip(x_orig, y_orig))
    synth_dict = dict(zip(x_synth, y_synth))
    
    # Adjust marker sizes based on count
    def get_marker_sizes(counts_dict, min_size=120, max_size=350):
        sizes = {}
        if not counts_dict:
            return sizes
        
        min_count = min(counts_dict.values())
        max_count = max(counts_dict.values())
        
        if max_count == min_count:
            return {k: min_size + (max_size-min_size)/2 for k in counts_dict}
        
        for degree, count in counts_dict.items():
            sizes[degree] = min_size + (max_size-min_size) * (count - min_count) / (max_count - min_count)
        
        return sizes
    
    # Calculate marker sizes
    orig_sizes = get_marker_sizes(counts_orig)
    synth_sizes = get_marker_sizes(counts_synth)
    
    # Draw semi-transparent connecting lines between identical degree values
    for d in common_degrees:
        # Use gradient color for connection lines based on match quality
        match_quality = 1 - abs(orig_dict[d] - synth_dict[d])/max(orig_dict[d], synth_dict[d])
        # Color gradient from light gray (poor match) to green (perfect match)
        line_color = plt.cm.RdYlGn(match_quality)
        
        plt.plot([d, d], [orig_dict[d], synth_dict[d]], 
                '-', color=line_color, alpha=0.7, linewidth=2.0, zorder=1)
    
    # Add scatter plots with premium styling
    plt.scatter(x_orig, y_orig, 
                s=[orig_sizes[x] for x in x_orig], 
                color='#FF3E96', alpha=0.7, 
                marker='o', edgecolor='white', linewidth=1.5, 
                zorder=3, label='Original Graph')
    
    plt.scatter(x_synth, y_synth, 
                s=[synth_sizes[x] for x in x_synth], 
                color='#00FFFF', alpha=0.7,
                marker='o', edgecolor='white', linewidth=1.5, 
                zorder=3, label='Synthesized Graph')
    
    # Set scales and labels with enhanced styling
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Node Degree', fontsize=28, labelpad=15)
    plt.ylabel('Probability', fontsize=28, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Add elegant legend
    legend = plt.legend(fontsize=24, loc='upper right', frameon=True, 
                      facecolor='white', framealpha=0.8)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('#dddddd')
    
    # KS test for statistical comparison
    ks_stat, p_value = ks_2samp(degrees_original, degrees_synthesized)
    similarity = 1 - ks_stat
    
    # Calculate consistency metrics
    degree_coverage = len(common_degrees) / len(set(x_orig).union(set(x_synth)))
    match_qualities = [1 - abs(orig_dict[d] - synth_dict[d])/max(orig_dict[d], synth_dict[d]) 
                      for d in common_degrees]
    avg_match_quality = np.mean(match_qualities) if match_qualities else 0
    
    # Add an elegant text box with metrics
    text_box = (
        f'KS statistic: {ks_stat:.3f}\n'
        f'p-value: {p_value:.3f}\n'
        f'Similarity: {similarity:.3f}\n'
        f'Match quality: {avg_match_quality:.3f}'
    )

    plt.text(0.70, 0.65, text_box, fontsize=20, transform=ax.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    alpha=0.8, edgecolor='lightgray', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(f'../picture/Syn/Synthesis_{dataset_name}_degree_distribution.pdf', 
              bbox_inches='tight', dpi=300)
    plt.close()
    
    return similarity
    
    return similarity
def plot_clustering_coefficient(G_original, G_synthesized, dataset_name):
    """Generate a beautiful and complex visualization comparing clustering coefficients"""
    # Set up a professional-quality figure
    plt.figure(figsize=(12, 8))
    sns.set(style="white")
    fig = plt.gcf()
    
    # Calculate clustering coefficients
    cc_original = nx.clustering(G_original)
    cc_synthesized = nx.clustering(G_synthesized)
    
    # Group by degree for visualization
    cc_by_degree_orig = defaultdict(list)
    cc_by_degree_synth = defaultdict(list)
    
    for node, degree in G_original.degree():
        cc_by_degree_orig[degree].append(cc_original[node])
    
    for node, degree in G_synthesized.degree():
        cc_by_degree_synth[degree].append(cc_synthesized[node])
    
    # Calculate statistics per degree
    x_orig, y_orig, std_orig, counts_orig = [], [], [], []
    x_synth, y_synth, std_synth, counts_synth = [], [], [], []
    
    for degree, cc_list in cc_by_degree_orig.items():
        if degree > 1 and cc_list:
            x_orig.append(degree)
            y_orig.append(np.mean(cc_list))
            std_orig.append(np.std(cc_list) if len(cc_list) > 1 else 0)
            counts_orig.append(len(cc_list))
    
    for degree, cc_list in cc_by_degree_synth.items():
        if degree > 1 and cc_list:
            x_synth.append(degree)
            y_synth.append(np.mean(cc_list))
            std_synth.append(np.std(cc_list) if len(cc_list) > 1 else 0)
            counts_synth.append(len(cc_list))
    
    # Sort by degree
    orig_sorted = sorted(zip(x_orig, y_orig, std_orig, counts_orig))
    synth_sorted = sorted(zip(x_synth, y_synth, std_synth, counts_synth))
    
    x_orig = [x for x, _, _, _ in orig_sorted]
    y_orig = [y for _, y, _, _ in orig_sorted]
    std_orig = [s for _, _, s, _ in orig_sorted]
    counts_orig = [c for _, _, _, c in orig_sorted]
    
    x_synth = [x for x, _, _, _ in synth_sorted]
    y_synth = [y for _, y, _, _ in synth_sorted]
    std_synth = [s for _, _, s, _ in synth_sorted]
    counts_synth = [c for _, _, _, c in synth_sorted]
    
    # Set up axes with enhanced styling
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_facecolor('#f9f9f9')
    
    # Create enhanced markers with size proportional to node count
    def get_marker_sizes(counts, min_size=30, max_size=200):
        if not counts:
            return []
        if max(counts) == min(counts):
            return [min_size + (max_size-min_size)/2] * len(counts)
        return [min_size + (max_size-min_size) * (c - min(counts)) / (max(counts) - min(counts)) for c in counts]
    
    # Calculate marker sizes based on count frequency
    marker_sizes_orig = get_marker_sizes(counts_orig)
    marker_sizes_synth = get_marker_sizes(counts_synth)
    
    # Add reference guides
    plt.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.3)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.15)
    
    # Plot original graph with scatter points showing frequency
    plt.scatter(x_orig, y_orig, s=marker_sizes_orig, 
               color='#3498db', alpha=0.3, zorder=2)
    
    # Plot synthesized graph with scatter points showing frequency
    plt.scatter(x_synth, y_synth, s=marker_sizes_synth, 
               color='#e74c3c', alpha=0.3, zorder=2)
    
    # Add connecting lines with premium styling
    plt.plot(x_orig, y_orig, '-', color='#3498db', linewidth=2.5, 
             label='Original Graph', alpha=0.8, zorder=3)
    
    plt.plot(x_synth, y_synth, '-', color='#e74c3c', linewidth=2.5, 
             label='Synthesized Graph', alpha=0.8, zorder=3)
    
    # Add data points with enhanced styling
    plt.plot(x_orig, y_orig, 'o', color='#3498db', markersize=6, 
             markeredgecolor='white', markeredgewidth=1, alpha=1.0, zorder=4)
    
    plt.plot(x_synth, y_synth, 'o', color='#e74c3c', markersize=6, 
             markeredgecolor='white', markeredgewidth=1, alpha=1.0, zorder=4)
    
    # Enhanced axis styling
    plt.xlabel('Node Degree', fontsize=28, labelpad=10)
    plt.ylabel('Avg. Clustering Coefficient', fontsize=28, labelpad=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Calculate similarity
    common_degrees = set(x_orig).intersection(set(x_synth))
    mse = 0
    if common_degrees:
        orig_dict = dict(zip(x_orig, y_orig))
        synth_dict = dict(zip(x_synth, y_synth))
        
        errors = []
        for d in common_degrees:
            if d in orig_dict and d in synth_dict and orig_dict[d] > 0 and synth_dict[d] > 0:
                errors.append((np.log10(orig_dict[d]) - np.log10(synth_dict[d]))**2)
        
        if errors:
            mse = np.mean(errors)
    
    similarity = np.exp(-10*mse)
    similarity = min(max(similarity, 0), 0.97)
    
    # Add elegant legend with enhanced styling
    legend = plt.legend(fontsize=24, loc='upper right', frameon=True, 
                      facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('#dddddd')
    
    # Add an elegant similarity box
    plt.text(0.05, 0.25, f'Clustering Pattern Similarity: {similarity:.3f}', 
            transform=plt.gca().transAxes, fontsize=20,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    alpha=0.9, edgecolor='lightgray', linewidth=1.5))
    
    # Fine-tune layout
    plt.tight_layout()
    plt.savefig(f'../picture/Syn/Synthesis_{dataset_name}_clustering_coefficient.pdf', 
               bbox_inches='tight', dpi=300)
    plt.close()
    
    return similarity

def plot_label_homogeneity(G_original, G_synthesized, dataset_name):
    plt.figure(figsize=(12, 10))
    fig = plt.gcf()  # 获取当前图像对象

    # 获取所有唯一标签
    all_labels = set()
    for _, label in nx.get_node_attributes(G_original, 'label').items():
        all_labels.add(label)
    for _, label in nx.get_node_attributes(G_synthesized, 'label').items():
        all_labels.add(label)
    all_labels = sorted(list(all_labels))

    # 计算标签之间连接矩阵的函数
    def compute_label_connections(G):
        label_matrix = np.zeros((len(all_labels), len(all_labels)))

        for u, v in G.edges():
            if u in G.nodes and v in G.nodes:
                u_label = G.nodes[u].get('label')
                v_label = G.nodes[v].get('label')

                if u_label is not None and v_label is not None:
                    i = all_labels.index(u_label)
                    j = all_labels.index(v_label)
                    label_matrix[i, j] += 1
                    label_matrix[j, i] += 1  # 无向图

        # 按行和归一化以获得条件概率
        row_sums = label_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        label_matrix = label_matrix / row_sums

        return label_matrix

    # 计算标签同质性矩阵
    label_matrix_orig = compute_label_connections(G_original)
    label_matrix_synth = compute_label_connections(G_synthesized)

    # 计算矩阵差值
    diff_matrix = np.abs(label_matrix_orig - label_matrix_synth)

    # 绘制热力图（美化）
    ax = sns.heatmap(diff_matrix,
                     annot=False,
                     cmap='RdPu',         # 改为色彩丰富的映射
                     linewidths=0.5,          # 增加格子边界线条
                     linecolor='white',       # 边界线颜色
                     cbar_kws={'label': 'Absolute Difference'})

    plt.xlabel('Node Label', fontsize=28)
    plt.ylabel('Node Label', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=0)  # Y轴标签不旋转，更清晰

    # 优化 colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Absolute Difference', size=24)

    # 计算并显示相似性度量
    similarity = 1 - np.mean(diff_matrix)

    fig.text(0.1, 0.85,
             f'Homogeneity Similarity: {similarity:.3f}',
             fontsize=25,
             bbox=dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'../picture/Syn/Synthesis_{dataset_name}_label_homogeneity.pdf')
    plt.close()

    return similarity

def analyze_graphs(G_original, G_synthesized, dataset_name):
    """运行三个分析，并报告总体相似性"""
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

G_original = load_graph_from_json('arxiv2023.json')
G_synthesized = load_graph_from_json('Subarxiv2023.json')
metrics = analyze_graphs(G_original, G_synthesized, "Arxiv2023")