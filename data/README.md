# Data-Limited Graph Dataset Creation

This repository contains the code for creating data-limited graph datasets as described in our paper "GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments".

## Overview

In graph machine learning research, evaluating algorithms in data-limited environments is crucial but challenging. This code implements the M-Preserving Graph Sampling algorithm, which creates realistic data-limited variants of graph datasets while preserving essential structural and semantic properties.

## Installation

```
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- networkx
- numpy
- scipy
- matplotlib
- seaborn
- community (python-louvain)

## Usage

### Basic Usage

```bash
Python data_limited_creator.py input_file.json output_file.json sampling_ratio
```

## Algorithm Details

The M-Preserving Graph Sampling algorithm works through these steps:

1. **Property Extraction**: Analyzes the original graph to extract key structural properties

2. **Community Detection**: Identifies communities using modularity optimization

3. **Attribute Partitioning**: Creates partitions based on node labels and manifold properties

4. **Stratified Sampling**: Samples nodes while maintaining class distribution

5. Multi-objective Selection

   : For each partition, selects nodes based on:

   - Node degree (network importance)
   - Community representation (structural diversity)
   - Bridge potential (connectivity preservation)

6. **Structural Refinement**: Iteratively replaces nodes to minimize topological distortion

## Example Datasets

The paper used this algorithm to create data-limited variants of six standard benchmarks:

| Original Dataset | Data-Limited Variant | # Nodes (Original) | # Nodes (Limited) | # Edges (Original) | # Edges (Limited) |
| ---------------- | -------------------- | ------------------ | ----------------- | ------------------ | ----------------- |
| Cora             | SubCora              | 2,708              | 1,354             | 5,278              | 2,486             |
| Citeseer         | SubCiteseer          | 3,186              | 1,274             | 4,225              | 1,360             |
| Wikics           | SubWikics            | 8,196              | 1,639             | 104,161            | 26,786            |
| History          | SubHistory           | 41,551             | 2,077             | 251,590            | 40,415            |
| Arxiv2023        | SubArxiv2023         | 46,198             | 2,309             | 38,863             | 3,119             |
| Children         | SubChildren          | 76,875             | 3,843             | 1,162,522          | 94,636            |

## Validation

To verify the quality of your data-limited graph, the repository includes visualization tools:

```python
from graph_analyzer import plot_degree_distribution, plot_clustering_coefficient, plot_label_homogeneity

# Compare original and data-limited graphs
plot_degree_distribution(original_graph, sub_graph, "dataset_name")
plot_clustering_coefficient(original_graph, sub_graph, "dataset_name")  
plot_label_homogeneity(original_graph, sub_graph, "dataset_name")
```

