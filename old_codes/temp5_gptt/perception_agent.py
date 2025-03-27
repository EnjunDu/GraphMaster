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
    Responsible for constructing the graph, performing community detection (with semantic similarity), sampling nodes, and generating environment reports.
    Incorporates textual semantic similarity into the Louvain community detection by using node text embeddings as edge weight components.
    """

    def __init__(self,
                 data_file: str,
                 llm_pipeline: TextGenerationPipeline,
                 max_new_tokens: int = 1024,
                 top_percent: float = 0.1,
                 sample_size: int = 30):
        """
        Initialize the perception agent.
        :param data_file: Path to JSON file containing graph nodes (with neighbors and attributes).
        :param llm_pipeline: LLM text generation pipeline for summarization and decisions.
        :param max_new_tokens: Max tokens for LLM generation (for prompts).
        :param top_percent: Percentage threshold (e.g., 0.1 for top 10%) for PPR-based sampling.
        :param sample_size: Maximum number of nodes to sample for augmentation.
        """
        self.data_file = data_file
        self.llm_pipeline = llm_pipeline
        self.max_new_tokens = max_new_tokens
        self.top_percent = top_percent
        self.sample_size = sample_size
        self.G = None                # NetworkX graph
        self.partition = {}          # Louvain community partition (node_id -> community_id)
        self.sorted_communities = [] # Community IDs sorted by size (desc)
        self.ppr_scores = {}         # Personalized PageRank scores for sampling
        # Dictionary to store node text embeddings (for semantic similarity)
        self.node_embeddings = {}

        # Build the graph and run initial community detection with semantic integration
        self._build_graph_from_data()
        self._run_louvain_partition()
        self._sort_communities_by_size()

    def generate_environment_report(self, require_label_distribution: bool = False, data_file: str = None) -> str:
        """
        Generate a structured environment report of the graph, including global stats, community distribution, and optional label distribution.
        If a new data_file is provided (for evaluating an enhanced graph), rebuilds the graph from that file.
        The report structure aligns with the paper's method section:
          - ρ_global: global graph statistics (nodes, edges, etc.)
          - ρ_class^c: class-level (label) distribution
          - ρ_comm^i: community-level distribution (sizes per community)
          - D_struct: structural distribution metrics (e.g., avg degree, density, clustering coefficient)
          - D_sem: semantic distribution metrics (e.g., label embedding variance or centroid distance)
        Returns the environment report as a JSON string.
        """
        main_logger = logging.getLogger("main_logger")

        # If a different data file is specified (e.g., for enhanced graph evaluation), rebuild the graph from that file.
        if data_file is not None and data_file != self.data_file:
            main_logger.info(f"[GraphPerceptionAgent] Rebuilding graph from {data_file} for environment report on new data")
            old_data_file = self.data_file
            self.data_file = data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            defer_restore = True  # flag to restore original graph after report
        else:
            defer_restore = False

        if self.G is None or self.G.number_of_nodes() == 0:
            return json.dumps({"error": "Graph is empty. No environment data available."})

        # Global graph statistics
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0  # average degree in undirected graph
        density = nx.density(self.G) if num_nodes > 1 else 0.0
        # Clustering coefficient (average clustering of nodes, ignoring 0 for isolated nodes if any)
        clustering_coeff = nx.average_clustering(self.G) if num_nodes > 0 else 0

        # Community distribution
        community_sizes = [len([n for n in self.partition if self.partition[n] == cid]) for cid in self.sorted_communities]
        community_distribution = {str(cid): size for cid, size in zip(self.sorted_communities, community_sizes)}

        report = {
            "GlobalStats": {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_degree": round(avg_degree, 3),
                "density": round(density, 5),
                "avg_clustering_coeff": round(clustering_coeff, 5)
            },
            "CommunityDistribution": community_distribution
        }

        if require_label_distribution:
            label_count = self._compute_label_distribution()
            report["LabelDistribution"] = label_count

        # Semantic distribution: for each label, we can compute a centroid of embeddings and average distance to centroid (as a measure of variance).
        # This provides insight into semantic coherence of each class.
        semantic_stats = {}
        if self.node_embeddings:
            # Group embeddings by label
            label_embeddings = {}
            for node_id, data in self.G.nodes(data=True):
                label = str(data.get('label', 'unknown'))
                if node_id in self.node_embeddings:
                    label_embeddings.setdefault(label, []).append(self.node_embeddings[node_id])
            for label, emb_list in label_embeddings.items():
                if emb_list:
                    emb_array = np.array(emb_list)
                    centroid = emb_array.mean(axis=0)
                    # Compute average Euclidean distance from centroid for this label
                    dists = np.linalg.norm(emb_array - centroid, axis=1)
                    semantic_stats[label] = {
                        "count": len(emb_list),
                        "avg_dist_to_centroid": float(round(np.mean(dists), 5))
                    }
        report["SemanticDistribution"] = semantic_stats

        # Optionally, get an LLM summary of structural characteristics (not required by new explicit structure but kept for context)
        try:
            summary_prompt = (
                f"The graph has {num_nodes} nodes and {num_edges} edges.\n"
                f"Communities (size distribution): {community_distribution}\n"
            )
            if require_label_distribution:
                summary_prompt += f"Label distribution: {report['LabelDistribution']}\n"
            summary_prompt += "Summarize the graph's structure and label balance in a few sentences."
            summary = self._call_generation(summary_prompt)
            report["Summary"] = summary.strip()
        except Exception as e:
            main_logger.warning(f"[GraphPerceptionAgent] Could not generate summary via LLM: {e}")
            report["Summary"] = ""

        # If we rebuilt the graph for this report, restore the original graph state
        if defer_restore:
            self.data_file = old_data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            main_logger.info(f"[GraphPerceptionAgent] Restored original graph from {old_data_file}")

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _compute_label_distribution(self) -> dict:
        """
        Scan all nodes and count occurrences of each 'label'.
        Returns a dictionary of label -> count.
        """
        label_count = {}
        for node_id, data in self.G.nodes(data=True):
            label = str(data.get('label', 'unknown'))
            label_count[label] = label_count.get(label, 0) + 1
        return label_count

    def _build_graph_from_data(self):
        """
        Construct the NetworkX graph G from the data_file (JSON list of nodes).
        Also computes a simple embedding for each node's text (for semantic similarity).
        This resets any existing graph and partition state.
        """
        main_logger = logging.getLogger("main_logger")
        self.G = nx.Graph()
        self.node_embeddings = {}  # reset embeddings

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)
            for node_info in nodes_data:
                node_id = node_info["node_id"]
                label = node_info.get("label", "unknown")
                mask = node_info.get("mask", "unknown")
                neighbors = node_info.get("neighbors", [])

                # Compute text embedding (placeholder: random vector, to be replaced by actual embedding in prod)
                text_content = node_info.get("text", "")
                embedding = self._compute_text_embedding(text_content)
                self.node_embeddings[node_id] = embedding

                # Add the node to the graph with its attributes
                self.G.add_node(node_id, label=label, mask=mask)
                # Add edges (undirected) for each neighbor (assuming neighbors listed are undirected connections)
                for nbr in neighbors:
                    # Avoid self-loop and ensure neighbor is added even if not yet encountered
                    if nbr is None or node_id == nbr:
                        continue
                    self.G.add_node(nbr)  # ensure neighbor exists as a node
                    self.G.add_edge(node_id, nbr)
            main_logger.info(f"[GraphPerceptionAgent] Graph built from {self.data_file} with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        except FileNotFoundError:
            main_logger.error(f"[GraphPerceptionAgent] data_file={self.data_file} not found. Graph is empty.")
            self.G = nx.Graph()
        except json.JSONDecodeError as e:
            main_logger.error(f"[GraphPerceptionAgent] Failed to parse JSON from {self.data_file} (error: {e}). Graph is empty.")
            self.G = nx.Graph()

    def _compute_text_embedding(self, text: str):
        """
        Compute a simple embedding vector for the given text.
        Here we use a fixed-size random vector seeded by text length for determinism.
        In a production setting, replace this with a real text embedding model (e.g., using sentence transformers).
        """
        rng = np.random.RandomState(len(text) + 2023)  # seed with text length for reproducibility
        return rng.rand(128)  # 128-dimensional random vector

    def _run_louvain_partition(self, gamma: float = 0.5):
        """
        Run a Louvain community detection algorithm with a custom edge weighting that blends structure and semantic similarity.
        gamma balances structure vs semantics (1.0 means only structure, 0.0 means only semantic similarity).
        Sets self.partition (node_id -> community_id).
        """
        main_logger = logging.getLogger("main_logger")
        if self.G is None or self.G.number_of_nodes() == 0:
            self.partition = {}
            main_logger.warning("[GraphPerceptionAgent] Graph is empty, skipping community detection.")
            return

        # Internal helper for cosine similarity between two embedding vectors
        def cosine_sim(vec1, vec2):
            if vec1 is None or vec2 is None:
                return 0.0
            dot_val = float(np.dot(vec1, vec2))
            norm1 = float(np.linalg.norm(vec1))
            norm2 = float(np.linalg.norm(vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_val / (norm1 * norm2)

        # Remove any existing 'weight' attributes on edges
        for (u, v) in self.G.edges():
            if "weight" in self.G[u][v]:
                del self.G[u][v]["weight"]

        # Compute new edge weights based on structure and semantic similarity
        for (u, v) in list(self.G.edges()):
            # structural_part = 1 for an existing edge
            structural_part = 1.0
            emb_u = self.node_embeddings.get(u)
            emb_v = self.node_embeddings.get(v)
            semantic_part = cosine_sim(emb_u, emb_v)
            # Combined weight: higher weight if nodes are connected and semantically similar
            w_uv = gamma * structural_part + (1.0 - gamma) * semantic_part
            self.G[u][v]["weight"] = w_uv

        # Run Louvain partition (community_louvain.best_partition) using our weights
        try:
            self.partition = community_louvain.best_partition(self.G, weight='weight', resolution=1.0)
            num_communities = max(self.partition.values()) + 1 if self.partition else 0
            main_logger.info(f"[GraphPerceptionAgent] Louvain found {num_communities} communities (gamma={gamma}).")
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error in Louvain community detection: {e}")
            self.partition = {}

    def _sort_communities_by_size(self):
        """
        Sort community IDs by the number of nodes in each community (descending order).
        Store the sorted list of community IDs in self.sorted_communities.
        """
        if not self.partition:
            self.sorted_communities = []
            return
        community_nodes = {}
        for node, cid in self.partition.items():
            community_nodes.setdefault(cid, []).append(node)
        self.sorted_communities = sorted(community_nodes.keys(), key=lambda c: len(community_nodes[c]), reverse=True)

    def decide_sampling(self, visualize: bool = False) -> list:
        """
        For semantic enhancement mode: use LLM to pick a smaller community to focus on, then sample top-PPR nodes.
        1) Summarize communities and have LLM choose a community ID from the smaller half (but not the absolute smallest) to ensure enough nodes.
        2) Compute Personalized PageRank (PPR) scores for that community's nodes against the whole graph.
        3) Sample the top `sample_size` nodes from the top `top_percent` PPR scorers.
        Optionally visualize the sampled nodes on the graph.
        """
        main_logger = logging.getLogger("main_logger")
        if not self.partition:
            main_logger.info("[GraphPerceptionAgent] No community partition available, cannot sample.")
            return []

        # Prepare community info for LLM prompt
        community_info = []
        community_nodes_dict = {}
        for node, cid in self.partition.items():
            community_nodes_dict.setdefault(cid, []).append(node)
        for rank, cid in enumerate(self.sorted_communities):
            size = len(community_nodes_dict[cid])
            community_info.append({"rank": rank, "community_id": cid, "num_nodes": size})

        # LLM prompt to choose a community
        prompt = (
            "You are a Graph Perception Agent.\n"
            "We performed Louvain community detection on a graph.\n"
            f"Community distribution (largest to smallest):\n{json.dumps(community_info, ensure_ascii=False, indent=2)}\n\n"
            "Choose one community from the smaller half (but not the absolute smallest) to focus on for semantic enhancement.\n"
            "Respond with a JSON object: {\"selected_community_id\": <id>} with no extra text."
        )
        raw_output = self._call_generation(prompt)
        selection = self._extract_json(raw_output)
        if "selected_community_id" in selection:
            try:
                selected_community_id = int(selection["selected_community_id"])
            except ValueError:
                selected_community_id = selection["selected_community_id"]
        else:
            main_logger.warning("[GraphPerceptionAgent] LLM did not provide a valid community selection. Using default heuristic.")
            # Default: choose the median community (to avoid largest and smallest extremes)
            if not self.sorted_communities:
                return []
            mid_point = len(self.sorted_communities) // 2
            selected_community_id = self.sorted_communities[mid_point]

        main_logger.info(f"[GraphPerceptionAgent] Selected community ID for sampling: {selected_community_id}")

        # Calculate PPR scores from the selected community
        self.ppr_scores = self._calculate_ppr_scores(selected_community_id)
        if not self.ppr_scores:
            main_logger.warning("[GraphPerceptionAgent] No PPR scores calculated (perhaps community too small). Using random sampling.")
            sample_size = min(self.sample_size, len(self.G.nodes))
            return random.sample(list(self.G.nodes), sample_size)

        # Sort nodes by PPR score (desc) and pick the top X%
        sorted_nodes = sorted(self.ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = max(1, int(len(sorted_nodes) * self.top_percent))
        top_nodes = [node for node, _ in sorted_nodes[:top_k]]
        # Randomly sample up to sample_size from these top nodes
        sample_size = min(self.sample_size, len(top_nodes))
        final_samples = random.sample(top_nodes, sample_size)
        main_logger.info(f"[GraphPerceptionAgent] Sampled {len(final_samples)} nodes from top {top_k} PPR nodes.")

        # Visualization (if enabled)
        if visualize and final_samples:
            self._visualize_sampling(final_samples)
        return final_samples

    def decide_label_sampling(self, target_label: int, label_target_size: int, visualize: bool = False) -> list:
        """
        For topological enhancement mode: decide which nodes to sample for augmentation.
        This focuses on nodes of the target label (to augment that label's subgraph).
        Strategy:
          - If current count for target_label >= label_target_size, nothing to sample (enough data).
          - If very few target_label nodes exist (<5), return all of them (small set).
          - Else, for each existing node of target_label (Train mask only), compute PPR relative to other target_label nodes and average scores.
          - Take the top top_percent of those nodes by PPR and sample up to sample_size from them.
        """
        main_logger = logging.getLogger("main_logger")
        if target_label is None or label_target_size <= 0:
            main_logger.warning("[GraphPerceptionAgent] Topological enhancement called without valid target_label or label_target_size.")
            return []

        # Identify existing nodes of the target label (only considering training nodes for augmentation).
        label_nodes = [n for n, attr in self.G.nodes(data=True) if str(attr.get('label')) == str(target_label) and attr.get('mask') == "Train"]
        current_label_count = len(label_nodes)
        main_logger.info(f"[GraphPerceptionAgent] Found {current_label_count} nodes with label={target_label} (mask='Train').")

        # If target label already meets or exceeds desired size, no sampling needed.
        if current_label_count >= label_target_size:
            main_logger.info("[GraphPerceptionAgent] Sufficient nodes for target label; skipping sampling.")
            return []

        # If very few nodes of target label, using them all (to bootstrap augmentation).
        if current_label_count <= 5:
            main_logger.info(f"[GraphPerceptionAgent] Only {current_label_count} nodes of label={target_label} exist, using all for augmentation.")
            if visualize and label_nodes:
                self._visualize_sampling(label_nodes)
            return label_nodes

        # Compute PPR scores focusing on target_label nodes
        temp_ppr_scores = {}
        for start_node in label_nodes:
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0
                ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)
                # Only consider PPR scores for target_label nodes
                for node in label_nodes:
                    temp_ppr_scores.setdefault(node, []).append(ppr.get(node, 0.0))
            except Exception as e:
                main_logger.warning(f"[GraphPerceptionAgent] PPR computation failed for node {start_node}: {e}")
                continue

        # Compute average PPR score for each target_label node
        avg_ppr_scores = {node: (sum(scores) / len(scores)) for node, scores in temp_ppr_scores.items() if scores}
        if avg_ppr_scores:
            sorted_nodes = sorted(avg_ppr_scores.items(), key=lambda x: x[1], reverse=True)
            top_k = max(1, int(len(sorted_nodes) * self.top_percent))
            top_nodes = [node for node, _ in sorted_nodes[:top_k]]
            sample_size = min(self.sample_size, len(top_nodes), label_target_size - current_label_count or len(top_nodes))
            sampled_nodes = random.sample(top_nodes, sample_size)
        else:
            # If PPR fails, fall back to random subset of target_label nodes
            sample_size = min(self.sample_size, len(label_nodes))
            sampled_nodes = random.sample(label_nodes, sample_size)

        main_logger.info(f"[GraphPerceptionAgent] For label {target_label}, sampled {len(sampled_nodes)} nodes for enhancement.")
        if visualize and sampled_nodes:
            self._visualize_sampling(sampled_nodes)
        return sampled_nodes

    def _visualize_sampling(self, sampled_nodes: list):
        """
        Visualize the sampled nodes on the graph (for debugging/analysis).
        Sampled nodes are highlighted in red, others in gray. If PPR scores are available for more nodes, color-code them in blue shades.
        """
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 10))
        # Draw all nodes in light gray
        nx.draw_networkx_nodes(self.G, pos, node_color="lightgray", node_size=50, alpha=0.6)
        # Draw sampled nodes in red
        nx.draw_networkx_nodes(self.G, pos, nodelist=sampled_nodes, node_color="red", node_size=100, alpha=0.9)
        # If PPR scores available for other nodes, draw those nodes in a blue gradient
        if self.ppr_scores:
            ppr_nodes = set(self.ppr_scores.keys()) - set(sampled_nodes)
            if ppr_nodes:
                ppr_values = [self.ppr_scores[node] for node in ppr_nodes]
                nx.draw_networkx_nodes(self.G, pos, nodelist=list(ppr_nodes),
                                       node_color=ppr_values, cmap=plt.cm.Blues,
                                       node_size=70, alpha=0.8)
                # Add a color bar to show PPR score range
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
                sm.set_array([min(ppr_values), max(ppr_values) if ppr_values else 0])
                plt.colorbar(sm, label="PPR Score")
        # Draw edges lightly
        nx.draw_networkx_edges(self.G, pos, width=0.5, alpha=0.3)
        # Label the sampled nodes with their IDs for clarity
        labels = {node: str(node) for node in sampled_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)
        plt.title("Sampled Nodes for Enhancement")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("ppr_sampling_visualization.png", dpi=300)
        plt.show()

    def _calculate_ppr_scores(self, community_id):
        """
        Calculate personalized PageRank (PPR) scores for each node in the graph, personalized to each node in the given community.
        We compute PPR starting from each node in the community (one at a time), and average the scores.
        Returns a dict of node -> average PPR score across all starting nodes in the community.
        """
        main_logger = logging.getLogger("main_logger")
        community_nodes = [node for node, cid in self.partition.items() if cid == community_id]
        if not community_nodes:
            main_logger.warning(f"[GraphPerceptionAgent] No nodes found in community {community_id}")
            return {}
        main_logger.info(f"[GraphPerceptionAgent] Calculating PPR scores for community {community_id} with {len(community_nodes)} nodes.")

        all_ppr_scores = {}
        for start_node in community_nodes:
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0
                ppr_result = nx.pagerank(self.G, alpha=0.85, personalization=personalization)
                for node, score in ppr_result.items():
                    all_ppr_scores.setdefault(node, []).append(score)
            except Exception as e:
                main_logger.warning(f"[GraphPerceptionAgent] Failed to calculate PPR for node {start_node}: {e}")
                continue

        avg_ppr_scores = {}
        for node, scores in all_ppr_scores.items():
            if scores:
                avg_ppr_scores[node] = sum(scores) / len(scores)
        main_logger.info(f"[GraphPerceptionAgent] Computed average PPR scores for {len(avg_ppr_scores)} nodes.")
        return avg_ppr_scores

    def _call_generation(self, prompt: str) -> str:
        """
        Use the LLM pipeline to generate text for a given prompt.
        (Used for summary generation or sampling decisions, with greedy decoding and moderate temperature.)
        """
        main_logger = logging.getLogger("main_logger")
        try:
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=1
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error during text generation: {e}")
            return ""

    def _extract_json(self, text: str) -> dict:
        """
        Extract a JSON object from the given text. Used to parse LLM outputs that contain JSON snippets.
        We attempt to find the first JSON object in the text.
        """
        main_logger = logging.getLogger("main_logger")
        json_pattern = r'\{[^{}]*\}'  # a simple pattern for a single-level JSON (naive, but should suffice for expected output)
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                main_logger.error(f"[GraphPerceptionAgent] Failed to parse JSON from LLM output: {json_str}")
        # If direct JSON extraction fails, try to find a numeric id (for selected_community_id as a fallback)
        match = re.search(r'"selected_community_id"\s*:\s*([0-9]+)', text)
        if match:
            return {"selected_community_id": match.group(1)}
        return {}
