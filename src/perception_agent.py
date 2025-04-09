import json
import random
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
from transformers import TextGenerationPipeline
import numpy as np
import re
import logging

class GraphPerceptionAgent:
    """
    Perception Agent - implements the retrieval component of the RAG paradigm:
    - Constructs the graph from data
    - Performs semantic-enriched community detection
    - Implements query-conditioned community identification
    - Provides mode-adaptive seed selection strategy
    - Executes hierarchical stochastic diffusion sampling
    - Generates comprehensive environment reports
    """

    def __init__(self,
                 data_file: str,
                 llm_pipeline: TextGenerationPipeline,
                 max_new_tokens: int = 1024,
                 top_percent: float = 0.1,
                 sample_size: int = 30):
        """
        :param data_file: JSON file with graph node information
        :param llm_pipeline: LLM text generation pipeline
        :param max_new_tokens: Maximum new tokens for generation
        :param top_percent: Percentage of top PPR nodes to consider
        :param sample_size: Number of nodes to sample
        """
        self.data_file = data_file
        self.llm_pipeline = llm_pipeline
        self.max_new_tokens = max_new_tokens
        self.top_percent = top_percent
        self.sample_size = sample_size
        self.G = None
        self.partition = {}
        self.sorted_communities = []
        self.ppr_scores = {}

        # Store node text embeddings
        self.node_embeddings = {}

        # Build graph and perform community detection
        self._build_graph_from_data()
        self._run_louvain_partition()
        self._sort_communities_by_size()

    def generate_environment_report(self, require_label_distribution=False, data_file=None) -> str:
        """
        Generates comprehensive environmental status report encapsulating multi-scale properties:
        R_t = (ρ_global, {ρ_class^c}_c=1^C, {ρ_comm^i}_i=1^|C|, D_struct, D_sem)
        
        :param require_label_distribution: Whether to include label distribution
        :param data_file: Optional - rebuild graph from this file for enhanced state report
        :return: Environmental report as JSON string
        """
        main_logger = logging.getLogger("main_logger")

        # If new data file specified, rebuild graph
        if data_file is not None and data_file != self.data_file:
            main_logger.info(f"[GraphPerceptionAgent] Rebuilding graph from {data_file} for enhanced state report")
            old_data_file = self.data_file
            self.data_file = data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            defer_restore = True
        else:
            defer_restore = False

        if self.G is None or self.G.number_of_nodes() == 0:
            return "Graph is empty. No environment data available."

        # 1. Global statistics (ρ_global)
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        global_stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "density": density
        }
        
        # Calculate additional global metrics
        try:
            # Compute average clustering coefficient
            clustering = nx.average_clustering(self.G)
            global_stats["clustering_coefficient"] = clustering
            
            # Estimate average shortest path length (for connected components)
            connected_components = list(nx.connected_components(self.G))
            if connected_components:
                largest_cc = max(connected_components, key=len)
                if len(largest_cc) > 1:
                    subgraph = self.G.subgraph(largest_cc)
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                    global_stats["avg_path_length"] = avg_path_length
                
            # Component size distribution
            component_sizes = [len(c) for c in connected_components]
            global_stats["connected_components"] = len(component_sizes)
            global_stats["largest_component_size"] = max(component_sizes) if component_sizes else 0
            global_stats["component_size_distribution"] = component_sizes
        except Exception as e:
            main_logger.warning(f"[GraphPerceptionAgent] Error computing some global metrics: {e}")
        
        # 2. Community-level properties (ρ_comm^i)
        community_stats = {}
        for comm_id in self.sorted_communities:
            comm_nodes = [n for n in self.partition if self.partition[n] == comm_id]
            comm_size = len(comm_nodes)
            comm_subgraph = self.G.subgraph(comm_nodes)
            
            comm_stats = {
                "size": comm_size,
                "internal_edges": comm_subgraph.number_of_edges(),
                "fraction_of_graph": comm_size / num_nodes if num_nodes > 0 else 0
            }
            
            # Calculate modularity contribution
            internal_edges = comm_subgraph.number_of_edges()
            total_degree = sum(self.G.degree(n) for n in comm_nodes)
            expected_edges = (total_degree ** 2) / (2 * num_edges) if num_edges > 0 else 0
            modularity_contribution = (internal_edges / num_edges) - (expected_edges / num_edges) if num_edges > 0 else 0
            comm_stats["modularity_contribution"] = modularity_contribution
            
            community_stats[str(comm_id)] = comm_stats
        
        # 3. Class-level properties (ρ_class^c)
        class_stats = {}
        if require_label_distribution:
            label_count = self._compute_label_distribution()
            
            for label, count in label_count.items():
                # Fix: Handle non-numeric labels properly
                if str(label).isdigit():
                    # For numeric labels
                    label_nodes = [n for n, attr in self.G.nodes(data=True) 
                                 if attr.get('label') == int(label)]
                else:
                    # For non-numeric labels (like "unknown")
                    label_nodes = [n for n, attr in self.G.nodes(data=True) 
                                 if attr.get('label') == label]
                
                label_subgraph = self.G.subgraph(label_nodes)
                
                class_stats[label] = {
                    "count": count,
                    "fraction": count / num_nodes if num_nodes > 0 else 0,
                    "internal_edges": label_subgraph.number_of_edges(),
                    "avg_degree": 2 * label_subgraph.number_of_edges() / count if count > 0 else 0
                }
                
                # Community distribution of this class
                class_comm_dist = {}
                for n in label_nodes:
                    if n in self.partition:
                        comm = self.partition[n]
                        class_comm_dist[str(comm)] = class_comm_dist.get(str(comm), 0) + 1
                
                class_stats[label]["community_distribution"] = class_comm_dist
        
        # 4. Structural distribution (D_struct)
        degree_dist = {}
        for d in dict(self.G.degree()).values():
            degree_dist[str(d)] = degree_dist.get(str(d), 0) + 1
        
        # 5. Semantic distribution (D_sem) - simplified placeholder
        semantic_dist = {"placeholder": "Semantic distribution analysis would go here"}
        
        # Combine all components into the report
        report = {
            "Graph": global_stats,
            "Communities": {
                "indices": self.sorted_communities,
                "sizes": [len([n for n in self.partition if self.partition[n] == comm_id])
                          for comm_id in self.sorted_communities],
                "distribution": {str(comm_id): len([n for n in self.partition if self.partition[n] == comm_id])
                                 for comm_id in self.sorted_communities},
                "statistics": community_stats
            },
            "StructuralDistribution": {
                "degree_distribution": degree_dist
            },
            "SemanticDistribution": semantic_dist
        }
        
        if require_label_distribution:
            report["LabelDistribution"] = label_count
            report["ClassStatistics"] = class_stats

        # Generate a summary using LLM
        summary_prompt = f"""You are a Graph Perception Agent.
We have analyzed a graph with {num_nodes} nodes and {num_edges} edges.
The community detection identified the following communities:
{json.dumps(report['Communities'], ensure_ascii=False, indent=2)}
"""

        if require_label_distribution:
            summary_prompt += f"And here is the label distribution:\n{json.dumps(report['LabelDistribution'], ensure_ascii=False, indent=2)}\n"

        summary_prompt += "Please summarize the graph's structural characteristics in a few sentences.\n"

        summary = self._call_generation(summary_prompt)
        report["Summary"] = summary.strip()

        # Restore original data file if needed
        if defer_restore:
            self.data_file = old_data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            main_logger.info(f"[GraphPerceptionAgent] Restored original graph from {old_data_file}")

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _compute_label_distribution(self):
        """
        Computes label distribution across the graph.
        """
        label_count = {}
        for node_id in self.G.nodes:
            label = self.G.nodes[node_id].get('label', 'unknown')
            label_count[str(label)] = label_count.get(str(label), 0) + 1
        return label_count

    def _build_graph_from_data(self):
        """
        Builds graph from data file, including text embeddings for semantic analysis.
        """
        main_logger = logging.getLogger("main_logger")

        self.G = nx.Graph()
        self.node_embeddings = {}  # Reset

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)

            for node_info in nodes_data:
                node_id = node_info["node_id"]
                label = node_info.get("label", "unknown")
                mask = node_info.get("mask", "unknown")
                neighbors = node_info.get("neighbors", [])

                # 1) Calculate text embedding
                text_content = node_info.get("text", "")
                embedding = self._compute_text_embedding(text_content)
                self.node_embeddings[node_id] = embedding

                # 2) Add node with attributes
                self.G.add_node(node_id, label=label, mask=mask)

                # 3) Add undirected edges
                for nbr in neighbors:
                    self.G.add_edge(node_id, nbr)

            main_logger.info(
                f"[GraphPerceptionAgent] Graph built from {self.data_file} "
                f"with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
            )
        except FileNotFoundError:
            main_logger.error(
                f"[GraphPerceptionAgent] data_file={self.data_file} not found. Graph is empty."
            )
            self.G = nx.Graph()
        except json.JSONDecodeError:
            main_logger.error(
                f"[GraphPerceptionAgent] Failed to parse JSON from {self.data_file}. Graph is empty."
            )
            self.G = nx.Graph()

    def _compute_text_embedding(self, text: str):
        """
        Computes text embedding vector.
        Production implementation would use a real text embedding model.
        """
        rng = np.random.RandomState(len(text) + 2023)  # Simple differentiation
        # 128-dimensional embedding vector
        return rng.rand(128)

    def _run_louvain_partition(self, gamma: float = 0.5):
        """
        Implements semantic-enriched modularity maximization:
        Q_sem = 1/2m ∑_{i,j} [A_{ij} - γ * k_i*k_j/2m - (1-γ) * d_sem(x_i, x_j)/∑_{l,m} d_sem(x_l, x_m)] * δ(c_i, c_j)
        
        :param gamma: Balance between topological (1) and semantic (0) information
        """
        main_logger = logging.getLogger("main_logger")

        if self.G is None or self.G.number_of_nodes() == 0:
            self.partition = {}
            main_logger.warning("[GraphPerceptionAgent] Graph is empty, skip community detection.")
            return

        # Calculate cosine similarity between embeddings
        def cosine_sim(vec1, vec2):
            if vec1 is None or vec2 is None:
                return 0.0
            dot_v = float(np.dot(vec1, vec2))
            norm1 = float(np.linalg.norm(vec1))
            norm2 = float(np.linalg.norm(vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_v / (norm1 * norm2)

        # Clear existing weights
        for (u, v) in self.G.edges():
            if "weight" in self.G[u][v]:
                del self.G[u][v]["weight"]

        # Calculate total semantic similarity for normalization
        total_semantic_sim = 0
        semantic_sims = {}
        for (u, v) in self.G.edges():
            emb_u = self.node_embeddings.get(u, None)
            emb_v = self.node_embeddings.get(v, None)
            sem_sim = cosine_sim(emb_u, emb_v)
            semantic_sims[(u, v)] = sem_sim
            total_semantic_sim += sem_sim

        # Normalize semantic similarities
        if total_semantic_sim > 0:
            for (u, v) in semantic_sims:
                semantic_sims[(u, v)] /= total_semantic_sim

        # Total number of edges (m in the formula)
        m = self.G.number_of_edges()
        
        # Calculate edge weights based on the semantic-enriched modularity
        for (u, v) in self.G.edges():
            # Structural component: A_{ij} - γ * k_i*k_j/2m
            k_u = self.G.degree(u)
            k_v = self.G.degree(v)
            structural_part = 1.0 - gamma * (k_u * k_v) / (2 * m) if m > 0 else 1.0
            
            # Semantic component: (1-γ) * d_sem(x_i, x_j)/∑_{l,m} d_sem(x_l, x_m)
            semantic_part = (1.0 - gamma) * semantic_sims.get((u, v), 0)
            
            # Combined weight
            w_uv = max(0.001, structural_part + semantic_part)  # Ensure positive weight
            self.G[u][v]["weight"] = w_uv

        # Run Louvain algorithm with custom weights
        try:
            self.partition = community_louvain.best_partition(self.G, weight='weight', resolution=1.0)
            num_communities = max(self.partition.values()) + 1 if self.partition else 0
            main_logger.info(f"[GraphPerceptionAgent] Semantic-enriched Louvain found {num_communities} communities with gamma={gamma}.")
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error in semantic-enriched Louvain: {e}")
            self.partition = {}

    def _sort_communities_by_size(self):
        """
        Sorts community IDs by size (number of nodes) in descending order.
        """
        if not self.partition:
            self.sorted_communities = []
            return

        community_nodes = {}
        for node, comm_id in self.partition.items():
            community_nodes.setdefault(comm_id, []).append(node)

        self.sorted_communities = sorted(
            community_nodes.keys(),
            key=lambda c: len(community_nodes[c]),
            reverse=True
        )

    def decide_sampling(self, visualize: bool = False) -> list:
        """
        Implements mode-adaptive seed selection for semantic enhancement:
        C_b = argmin_i |C_i| * (1 + μ * Var({x_j : v_j ∈ C_i}))
        
        Followed by hierarchical stochastic diffusion sampling.
        """
        main_logger = logging.getLogger("main_logger")

        if not self.partition:
            main_logger.info("[GraphPerceptionAgent] No partition available, so no sampling.")
            return []

        # 1) Collect community information
        community_info = []
        community_nodes_dict = {}
        for node, cid in self.partition.items():
            community_nodes_dict.setdefault(cid, []).append(node)

        # Calculate semantic variance within each community
        community_semantic_variance = {}
        for comm_id, nodes in community_nodes_dict.items():
            if len(nodes) <= 1:
                community_semantic_variance[comm_id] = 0
                continue
                
            # Extract embeddings for community nodes
            embeddings = [self.node_embeddings.get(node, None) for node in nodes]
            # Filter out None values
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if not valid_embeddings:
                community_semantic_variance[comm_id] = 0
                continue
                
            # Calculate variance of embeddings
            variance = np.mean([np.var(emb) for emb in valid_embeddings]) if valid_embeddings else 0
            community_semantic_variance[comm_id] = variance

        # Calculate community selection score according to the formula
        # C_b = argmin_i |C_i| * (1 + μ * Var({x_j : v_j ∈ C_i}))
        mu = 0.5  # Weight for semantic variance importance
        community_scores = {}
        
        for comm_id, nodes in community_nodes_dict.items():
            size = len(nodes)
            variance = community_semantic_variance.get(comm_id, 0)
            score = size * (1 + mu * variance)
            community_scores[comm_id] = score
            
            community_info.append({
                "community_id": comm_id,
                "num_nodes": size,
                "semantic_variance": variance,
                "selection_score": score
            })

        # Find community with minimum score (optimal for semantic enhancement)
        selected_community_id = min(community_scores.items(), key=lambda x: x[1])[0]
        
        main_logger.info(f"[GraphPerceptionAgent] Selected community ID: {selected_community_id} for semantic enhancement")
        main_logger.info(f"[GraphPerceptionAgent] Community selection score: {community_scores[selected_community_id]:.4f}")

        # 3) Calculate PPR scores using hierarchical stochastic diffusion
        self.ppr_scores = self._calculate_ppr_scores(selected_community_id)

        if not self.ppr_scores:
            main_logger.warning("[GraphPerceptionAgent] No PPR scores calculated. Falling back to random sampling.")
            # If PPR calculation fails, randomly sample from all nodes
            sample_size = min(self.sample_size, len(self.G.nodes))
            return random.sample(list(self.G.nodes), sample_size)

        # 4) Sort nodes by PPR score
        sorted_nodes = sorted(self.ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # 5) Implement hierarchical stochastic diffusion sampling
        # - Get top K% nodes by PPR score
        top_percent = self.top_percent
        top_k = max(1, int(len(sorted_nodes) * top_percent))
        top_nodes = [node for node, score in sorted_nodes[:top_k]]
        
        # - Apply stochastic sampling with probability proportional to PPR score
        beta = 2.0  # Control parameter for stochasticity
        sampled_nodes = []
        
        for node, score in sorted_nodes[:top_k]:
            # Probability proportional to normalized PPR score
            prob = min(1.0, beta * score / sorted_nodes[0][1])
            if random.random() < prob:
                sampled_nodes.append(node)
        
        # Ensure diversity by including at least one node from each major community
        major_communities = self.sorted_communities[:min(3, len(self.sorted_communities))]
        for comm_id in major_communities:
            if comm_id == selected_community_id:
                continue  # Already sampled from this community
                
            comm_nodes = [n for n, c in self.partition.items() if c == comm_id]
            if comm_nodes:
                # Add a random node from this community
                sampled_nodes.append(random.choice(comm_nodes))
        
        # Limit to sample_size
        if len(sampled_nodes) > self.sample_size:
            sampled_nodes = random.sample(sampled_nodes, self.sample_size)
        elif len(sampled_nodes) == 0:
            # If no nodes were sampled (unlikely), take top nodes by PPR score
            sampled_nodes = [node for node, _ in sorted_nodes[:min(self.sample_size, len(sorted_nodes))]]
        
        main_logger.info(f"[GraphPerceptionAgent] Selected {len(sampled_nodes)} nodes using hierarchical stochastic diffusion sampling")

        # 6) Visualize if requested
        if visualize and sampled_nodes:
            self._visualize_sampling(sampled_nodes)

        return sampled_nodes

    def decide_label_sampling(self, target_label: int, label_target_size: int, visualize: bool = False) -> list:
        """
        Selects nodes from minority classes for topological enhancement:
        C_b = {v_j ∈ V_train : y_j = argmax_c φ_imbal(c)}
        
        where φ_imbal(c) = max_{c'} |V_{c'}|/|V_c| quantifies class imbalance.
        """
        main_logger = logging.getLogger("main_logger")
        if not target_label or label_target_size <= 0:
            main_logger.warning("[GraphPerceptionAgent] Topological enhancement but invalid target_label or label_target_size.")
            return []

        # Find all nodes with the target label
        label_nodes = [n for n, attr in self.G.nodes(data=True)
                       if attr.get('label') == target_label and attr.get('mask') == "Train"]
        current_label_count = len(label_nodes)

        main_logger.info(f"[GraphPerceptionAgent] Found {current_label_count} nodes with label={target_label}.")

        # Calculate class imbalance factor
        label_counts = self._compute_label_distribution()
        max_label_count = max(int(count) for count in label_counts.values())
        imbalance_factor = max_label_count / (current_label_count + 1e-6)  # Avoid division by zero
        
        main_logger.info(f"[GraphPerceptionAgent] Imbalance factor for label={target_label}: {imbalance_factor:.2f}")

        # If already have enough nodes, no need to sample
        if current_label_count >= label_target_size:
            main_logger.info("[GraphPerceptionAgent] We already have enough nodes for that label.")
            return []

        # If target label has very few nodes, use all of them
        if len(label_nodes) <= 5:
            main_logger.info(f"[GraphPerceptionAgent] Too few nodes ({len(label_nodes)}) with label={target_label}, using all of them.")
            if visualize and label_nodes:
                self._visualize_sampling(label_nodes)
            return label_nodes

        # Set teleportation vector to focus on target label nodes
        personalization = {node: 0.0 for node in self.G.nodes()}
        for node in label_nodes:
            personalization[node] = 1.0 / len(label_nodes)
        
        # Calculate PPR with teleportation vector focused on target label nodes
        try:
            ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)
        except:
            main_logger.warning(f"[GraphPerceptionAgent] PageRank failed, using standard sampling.")
            # Fall back to random sampling
            sample_size = min(self.sample_size, len(label_nodes))
            return random.sample(label_nodes, sample_size)
        
        # Sort nodes by PPR score, but only consider nodes with target label
        sorted_nodes = [(node, score) for node, score in 
                        sorted(ppr.items(), key=lambda x: x[1], reverse=True)
                        if node in label_nodes]
        
        # Apply hierarchical stochastic diffusion sampling
        beta = 2.0  # Control parameter for stochasticity
        sampled_nodes = []
        
        for node, score in sorted_nodes:
            # Probability proportional to normalized PPR score
            prob = min(1.0, beta * score / sorted_nodes[0][1]) if sorted_nodes else 0
            if random.random() < prob:
                sampled_nodes.append(node)
        
        # Limit to sample_size
        if not sampled_nodes:
            # If no nodes selected, fall back to top nodes
            sampled_nodes = [node for node, _ in sorted_nodes[:self.sample_size]]
        elif len(sampled_nodes) > self.sample_size:
            sampled_nodes = random.sample(sampled_nodes, self.sample_size)
        
        main_logger.info(f"[GraphPerceptionAgent] For label={target_label}, sampled {len(sampled_nodes)} nodes for enhancement.")

        if visualize and sampled_nodes:
            self._visualize_sampling(sampled_nodes)

        return sampled_nodes

    def _visualize_sampling(self, sampled_nodes):
        """
        Visualizes sampled nodes with PPR score information.
        """
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 10))

        # Draw all nodes (gray)
        nx.draw_networkx_nodes(self.G, pos, node_color="lightgray", node_size=50, alpha=0.6)

        # Draw sampled nodes (red)
        nx.draw_networkx_nodes(self.G, pos, nodelist=sampled_nodes, node_color="red", node_size=100, alpha=0.9)

        # If PPR scores available, use color intensity to show scores
        if self.ppr_scores:
            ppr_nodes = set(self.ppr_scores.keys()) - set(sampled_nodes)
            if ppr_nodes:
                ppr_values = [self.ppr_scores[node] for node in ppr_nodes]
                nx.draw_networkx_nodes(self.G, pos, nodelist=list(ppr_nodes),
                                       node_color=ppr_values, cmap=plt.cm.Blues,
                                       node_size=70, alpha=0.8)

                # Add color bar showing PPR score range
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
                sm.set_array([min(ppr_values), max(ppr_values)])
                plt.colorbar(sm, label="PPR Score")

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, width=0.5, alpha=0.3)

        # Add labels for sampled nodes
        node_labels = {node: node for node in sampled_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8)

        plt.title("Sampled Nodes with PPR Scores Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("ppr_sampling_visualization.png", dpi=300)
        plt.show()

    def _calculate_ppr_scores(self, community_id):
        """
        Implements hierarchical stochastic diffusion using Personalized PageRank.
        π^(k+1) = αv + (1-α)W^T π^(k)
        """
        main_logger = logging.getLogger("main_logger")

        # Get all nodes in the specified community
        community_nodes = [node for node, comm in self.partition.items() if comm == community_id]

        if not community_nodes:
            main_logger.warning(f"[GraphPerceptionAgent] No nodes found in community {community_id}")
            return {}

        main_logger.info(f"[GraphPerceptionAgent] Calculating PPR scores for {len(community_nodes)} nodes in community {community_id}")

        # Set up teleportation vector focused on community nodes
        personalization = {node: 0.0 for node in self.G.nodes()}
        for node in community_nodes:
            personalization[node] = 1.0 / len(community_nodes)

        # Run personalized PageRank
        try:
            ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)
            main_logger.info(f"[GraphPerceptionAgent] Calculated PPR scores for {len(ppr)} nodes")
            return ppr
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error calculating PPR: {e}")
            return {}

    def _call_generation(self, prompt: str) -> str:
        """
        Calls LLM for text generation with error handling.
        """
        main_logger = logging.getLogger("main_logger")
        try:
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.90
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error during text generation: {e}")
            return "Error generating content. Please check the logs for details."

    def _extract_json(self, text: str) -> dict:
        """
        Extracts JSON data from LLM-generated text.
        """
        main_logger = logging.getLogger("main_logger")
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, text)

        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                main_logger.error(f"[GraphPerceptionAgent] Failed to parse JSON: {json_str}")

        # Try again looking specifically for "selected_community_id"
        comm_id_pattern = r'"selected_community_id"\s*:\s*(\d+)'
        match = re.search(comm_id_pattern, text)
        if match:
            return {"selected_community_id": match.group(1)}

        return {}