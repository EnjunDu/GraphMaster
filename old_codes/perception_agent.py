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
    Responsible for graph construction, community detection, node sampling using PPR scores, and generating environmental status reports.
    """

    def __init__(self,
                 data_file: str,
                 llm_pipeline: TextGenerationPipeline,
                 max_new_tokens: int = 1024):
        """
        :param data_file: JSON file path for storing graph node information (including neighbors)
        :param llm_pipeline: text generation pipeline for large language models
        :param max_new_tokens: maximum number of tokens that can be used during generation
        """
        self.data_file = data_file
        self.llm_pipeline = llm_pipeline
        self.max_new_tokens = max_new_tokens

        self.G = None
        self.partition = {}
        self.sorted_communities = []
        self.ppr_scores = {}

        # Build graph and perform community detection and ranking
        self._build_graph_from_data()
        self._run_louvain_partition()
        self._sort_communities_by_size()

    def generate_environment_report(self, require_label_distribution=False, data_file=None) -> str:
        """
        Generate an environment status report (community distribution overview, graph structure).
        If require_label_distribution=True, the number of nodes for each label is additionally counted and put into the report.

        :param require_label_distribution: Whether label distribution needs to be included
        :param data_file: Optional parameter, if provided, the graph is rebuilt from this file to evaluate the enhanced status
        :return: JSON string of the environment report
        """
        main_logger = logging.getLogger("main_logger")

        # If a new data file is specified, the graph is rebuilt.
        if data_file is not None and data_file != self.data_file:
            main_logger.info(f"[GraphPerceptionAgent] Rebuilding graph from {data_file} for enhanced state report")
            old_data_file = self.data_file
            self.data_file = data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            # Restore original data files after report generation
            defer_restore = True
        else:
            defer_restore = False

        if self.G is None or self.G.number_of_nodes() == 0:
            return "Graph is empty. No environment data available."

        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        report = {
            "Graph": {"num_nodes": num_nodes, "num_edges": num_edges},
            "Communities": {
                "indices": self.sorted_communities,
                "sizes": [
                    len([n for n in self.partition if self.partition[n] == comm_id])
                    for comm_id in self.sorted_communities
                ],
                "distribution": {
                    str(comm_id): len([n for n in self.partition if self.partition[n] == comm_id])
                    for comm_id in self.sorted_communities
                }
            }
        }

        if require_label_distribution:
            label_count = self._compute_label_distribution()
            report["LabelDistribution"] = label_count

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

        # If you need to restore the original data file, restore it here
        if defer_restore:
            self.data_file = old_data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            main_logger.info(f"[GraphPerceptionAgent] Restored original graph from {old_data_file}")

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _compute_label_distribution(self):
        """
        Scan all nodes and count the number of times their 'label' appears
        """
        label_count = {}
        for node_id in self.G.nodes:
            label = self.G.nodes[node_id].get('label', 'unknown')
            label_count[label] = label_count.get(label, 0) + 1
        return label_count

    def _calculate_ppr_scores(self, community_id):
        """
        Calculate the personalized PageRank score for each node in the specified community and take the average

        :param community_id: community ID
        :return: dictionary of average PPR scores corresponding to the node
        """
        main_logger = logging.getLogger("main_logger")

        # Get all nodes in a specified community
        community_nodes = [node for node, comm in self.partition.items() if comm == community_id]

        if not community_nodes:
            main_logger.warning(f"[GraphPerceptionAgent] No nodes found in community {community_id}")
            return {}

        main_logger.info(f"[GraphPerceptionAgent] Calculating PPR scores for {len(community_nodes)} nodes in community {community_id}")

        # Calculate PPR once for each community node
        all_ppr_scores = {}
        for start_node in community_nodes:
            # Use NetworkX's pagerank function to calculate personalized PageRank
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0

                # alpha is the damping coefficient, usually set to 0.85
                ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)

                # Accumulate the PPR score of each node
                for node, score in ppr.items():
                    all_ppr_scores.setdefault(node, []).append(score)
            except:
                main_logger.warning(f"[GraphPerceptionAgent] Failed to calculate PPR for node {start_node}")
                continue

        # Calculate the average PPR score for each node
        avg_ppr_scores = {}
        for node, scores in all_ppr_scores.items():
            if scores:  # Make sure you have scores before calculating the average
                avg_ppr_scores[node] = sum(scores) / len(scores)

        main_logger.info(f"[GraphPerceptionAgent] Calculated average PPR scores for {len(avg_ppr_scores)} nodes")
        return avg_ppr_scores

    def decide_sampling(self, visualize: bool = False) -> list:
        """
        Select a smaller community through LLM, calculate the PPR score, and sample the top 10% nodes with the highest PPR score.
        Used in semantic enhancement mode.
        """
        main_logger = logging.getLogger("main_logger")

        if not self.partition:
            main_logger.info("[GraphPerceptionAgent] No partition available, so no sampling.")
            return []

        # 1) Collect community distribution information
        community_info = []
        community_nodes_dict = {}
        for node, cid in self.partition.items():
            community_nodes_dict.setdefault(cid, []).append(node)

        for rank, comm_id in enumerate(self.sorted_communities):
            size = len(community_nodes_dict[comm_id])
            community_info.append({
                "rank": rank,
                "community_id": comm_id,
                "num_nodes": size
            })

        # 2) Use LLM to choose a smaller community as a starting point
        prompt = f"""You are a Graph Perception Agent.
We have performed Louvain community detection on a graph.
Below is the distribution of communities (from largest to smallest):
{json.dumps(community_info, ensure_ascii=False, indent=2)}

Your task is to select a community with fewer nodes as the starting community for semantic enhancement.
We want to focus on smaller communities that would benefit from data augmentation.

IMPORTANT: You MUST return the result as a JSON object with EXACTLY the following key: "selected_community_id".
The community you select should be in the smaller half of all communities, but not the absolute smallest to ensure enough data for calculating meaningful PPR scores.
Do not include any explanations, extra text, or additional keys.
"""

        raw_output = self._call_generation(prompt)
        community_selection = self._extract_json(raw_output)

        if "selected_community_id" not in community_selection:
            main_logger.warning("[GraphPerceptionAgent] LLM did not provide a valid community selection. Selecting a default smaller community.")
            # By default, a community in the second half of the sort is selected (avoid selecting the smallest one)
            mid_point = len(self.sorted_communities) // 2
            selected_community_id = self.sorted_communities[mid_point]
        else:
            selected_community_id = int(community_selection["selected_community_id"])

        main_logger.info(f"[GraphPerceptionAgent] Selected community ID: {selected_community_id}")

        # 3) Calculate the PPR score for the selected community
        self.ppr_scores = self._calculate_ppr_scores(selected_community_id)

        if not self.ppr_scores:
            main_logger.warning("[GraphPerceptionAgent] No PPR scores calculated. Falling back to random sampling.")
            # If the PPR calculation fails, randomly select some nodes from all nodes.
            sample_size = min(30, len(self.G.nodes))
            return random.sample(list(self.G.nodes), sample_size)

        # 4) Sort nodes by PPR score
        sorted_nodes = sorted(self.ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # 5) Randomly sample from the top 10% of nodes with the highest PPR scores
        top_percent = 0.1  # Top 10%
        top_k = max(1, int(len(sorted_nodes) * top_percent))
        top_nodes = [node for node, _ in sorted_nodes[:top_k]]

        # The number of samples should not exceed 30 nodes.
        sample_size = min(30, len(top_nodes))
        final_samples = random.sample(top_nodes, sample_size)

        main_logger.info(f"[GraphPerceptionAgent] Selected {sample_size} nodes from top {top_k} nodes by PPR score")

        # 6) Visualize sampling results if desired
        if visualize and final_samples:
            self._visualize_sampling(final_samples)

        return final_samples

    def decide_label_sampling(self, target_label: int, label_target_size: int, visualize: bool = False) -> list:
        """
        When the user selects the 'topological' mode, it is used to decide which nodes need to be sampled for subsequent enhancement.
        Here, the PPR score is also used to select nodes with high correlation with the target label.
        """
        main_logger = logging.getLogger("main_logger")
        if not target_label or label_target_size <= 0:
            main_logger.warning("[GraphPerceptionAgent] topological enhancement but invalid target_label or label_target_size.")
            return []

        # Find all nodes that match the label
        label_nodes = [n for n, attr in self.G.nodes(data=True)
                       if attr.get('label') == target_label and attr.get('mask') == "Train"]
        current_label_count = len(label_nodes)

        main_logger.info(f"[GraphPerceptionAgent] Found {current_label_count} nodes with label={target_label}.")

        # If the target has been achieved, no sampling is required
        if current_label_count >= label_target_size:
            main_logger.info("[GraphPerceptionAgent] We already have enough nodes for that label.")
            return []

        # If there are too few nodes with the target label, the calculation of PPR may be unstable, so all these nodes are returned directly.
        if len(label_nodes) <= 5:
            main_logger.info(f"[GraphPerceptionAgent] Too few nodes ({len(label_nodes)}) with label={target_label}, using all of them.")
            if visualize and label_nodes:
                self._visualize_sampling(label_nodes)
            return label_nodes

        # Calculate the PPR score for the target label node
        temp_ppr_scores = {}
        for start_node in label_nodes:
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0
                ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)

                # Only focus on the PPR scores of nodes with the same label
                for node in label_nodes:
                    temp_ppr_scores.setdefault(node, []).append(ppr[node])
            except:
                continue

        # Calculating the Average PPR Score
        avg_ppr_scores = {}
        for node, scores in temp_ppr_scores.items():
            if scores:
                avg_ppr_scores[node] = sum(scores) / len(scores)

        # Sort nodes by PPR score
        if avg_ppr_scores:
            sorted_nodes = sorted(avg_ppr_scores.items(), key=lambda x: x[1], reverse=True)
            # Take the top 10% of nodes
            top_percent = 0.1
            top_k = max(1, int(len(sorted_nodes) * top_percent))
            top_nodes = [node for node, _ in sorted_nodes[:top_k]]

            # The number of samples should not exceed 20 nodes.
            sample_size = min(20, len(top_nodes))
            sampled_nodes = random.sample(top_nodes, sample_size)
        else:
            # If the PPR calculation fails, random sampling
            sample_size = min(20, len(label_nodes))
            sampled_nodes = random.sample(label_nodes, sample_size)

        main_logger.info(f"[GraphPerceptionAgent] For label={target_label}, we sample {len(sampled_nodes)} nodes for enhancement.")

        if visualize and sampled_nodes:
            self._visualize_sampling(sampled_nodes)

        return sampled_nodes

    def _visualize_sampling(self, sampled_nodes):
        """
        Visualize the sampled nodes, including PPR score information.
        """
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 10))

        # Draw all nodes (gray)
        nx.draw_networkx_nodes(self.G, pos, node_color="lightgray", node_size=50, alpha=0.6)

        # Draw the sampling node (red)
        nx.draw_networkx_nodes(self.G, pos, nodelist=sampled_nodes, node_color="red", node_size=100, alpha=0.9)

        # If there is a PPR score, the color depth will indicate the PPR score.
        if self.ppr_scores:
            # Set color only for nodes with PPR score
            ppr_nodes = set(self.ppr_scores.keys()) - set(sampled_nodes)
            if ppr_nodes:
                ppr_values = [self.ppr_scores[node] for node in ppr_nodes]
                nx.draw_networkx_nodes(self.G, pos, nodelist=list(ppr_nodes),
                                       node_color=ppr_values, cmap=plt.cm.Blues,
                                       node_size=70, alpha=0.8)

                # Add a color bar to show the PPR score range
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
                sm.set_array([min(ppr_values), max(ppr_values)])
                plt.colorbar(sm, label="PPR Score")

        # draw edges
        nx.draw_networkx_edges(self.G, pos, width=0.5, alpha=0.3)

        # Add labels to sampling nodes
        node_labels = {node: node for node in sampled_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8)

        plt.title("Sampled Nodes with PPR Scores Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("ppr_sampling_visualization.png", dpi=300)
        plt.show()

    def _call_generation(self, prompt: str) -> str:
        outputs = self.llm_pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=1.0
        )
        return outputs[0]["generated_text"].strip()

    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON data from LLM generated text
        """
        main_logger = logging.getLogger("main_logger")
        # Trying to find a JSON object using regular expression
        json_pattern = r'\{[^{}]*\}'
        match = re.search(json_pattern, text)

        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                main_logger.error(f"[GraphPerceptionAgent] Failed to parse JSON: {json_str}")

        # If the above method fails, try looking for "selected_community_id" directly
        comm_id_pattern = r'"selected_community_id"\s*:\s*(\d+)'
        match = re.search(comm_id_pattern, text)
        if match:
            return {"selected_community_id": match.group(1)}

        return {}

    def _build_graph_from_data(self):
        """
        Load data from data_file and construct an undirected graph G based on node_id and neighbors.
        """
        main_logger = logging.getLogger("main_logger")

        self.G = nx.Graph()
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)

            # Add Node + Edge
            for node_info in nodes_data:
                node_id = node_info["node_id"]
                label = node_info.get("label", "unknown")
                mask = node_info.get("mask", "unknown")
                neighbors = node_info.get("neighbors", [])

                # Adding nodes and their attributes
                self.G.add_node(node_id, label=label, mask=mask)

                # Add edge (undirected edge)
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

    def _run_louvain_partition(self):
        """
        Performs Louvain community detection on the graph and records it in self.partition.
        """
        main_logger = logging.getLogger("main_logger")

        if self.G and self.G.number_of_nodes() > 0:
            try:
                self.partition = community_louvain.best_partition(self.G)
                num_communities = max(self.partition.values()) + 1 if self.partition else 0
                main_logger.info(f"[GraphPerceptionAgent] Louvain detected {num_communities} communities.")
            except Exception as e:
                main_logger.error(f"[GraphPerceptionAgent] Error in Louvain community detection: {e}")
                self.partition = {}
        else:
            self.partition = {}

    def _sort_communities_by_size(self):
        """
        Sort the community IDs in descending order according to the number of nodes in each community and store them in self.sorted_communities.
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