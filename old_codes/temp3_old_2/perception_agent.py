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
    负责图的构建、社区检测，利用PPR分数进行节点采样，并生成环境状态报告。
    现在额外支持在 Louvain 过程中加入文本语义相似度的混合边权。
    """

    def __init__(self,
                 data_file: str,
                 llm_pipeline: TextGenerationPipeline,
                 max_new_tokens: int = 1024,
                 top_percent: float = 0.1,
                 sample_size: int = 30):
        """
        :param data_file: 存储图节点信息 (包括 neighbors) 的 JSON 文件路径
        :param llm_pipeline: 大语言模型的文本生成 pipeline
        :param max_new_tokens: 生成时可使用的最大 tokens 数
        :param top_percent: 采样时使用的PPR分数的百分比
        :param sample_size: 采样时使用的节点数量
        """
        self.data_file = data_file
        self.llm_pipeline = llm_pipeline
        self.max_new_tokens = max_new_tokens
        self.top_percent = top_percent
        self.G = None
        self.sample_size = sample_size
        self.partition = {}
        self.sorted_communities = []
        self.ppr_scores = {}

        # 用于存放节点对应的文本向量
        self.node_embeddings = {}

        # 构建图并执行社区检测与排序
        self._build_graph_from_data()
        # 注意：_run_louvain_partition 里会融合语义相似度
        self._run_louvain_partition()
        self._sort_communities_by_size()

    def generate_environment_report(self, require_label_distribution=True, data_file=None) -> str:
        """
        生成环境状态报告（社区分布概况、图结构）。
        如果 require_label_distribution=True，则额外统计各 label 的节点数量并放入report。

        :param require_label_distribution: 是否需要包含标签分布
        :param data_file: 可选参数，如果提供则从该文件重新构建图，用于评估增强后的状态
        :return: 环境报告的JSON字符串
        """
        main_logger = logging.getLogger("main_logger")

        # 如果指定了新的数据文件，则重新构建图
        if data_file is not None and data_file != self.data_file:
            main_logger.info(f"[GraphPerceptionAgent] Rebuilding graph from {data_file} for enhanced state report")
            old_data_file = self.data_file
            self.data_file = data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            # 报告生成后恢复原始数据文件
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

        # 如果需要恢复原始数据文件，则在这里恢复
        if defer_restore:
            self.data_file = old_data_file
            self._build_graph_from_data()
            self._run_louvain_partition()
            self._sort_communities_by_size()
            main_logger.info(f"[GraphPerceptionAgent] Restored original graph from {old_data_file}")

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _compute_label_distribution(self):
        """
        扫描所有节点，统计其 'label' 出现次数
        """
        label_count = {}
        for node_id in self.G.nodes:
            label = self.G.nodes[node_id].get('label', 'unknown')
            label_count[label] = label_count.get(label, 0) + 1
        return label_count

    def _build_graph_from_data(self):
        """
        从 self.data_file 中加载数据，并依据 node_id 和 neighbors 构建无向图 G。
        同时为每个节点计算文本向量 self.node_embeddings[node_id]。
        """
        main_logger = logging.getLogger("main_logger")

        self.G = nx.Graph()
        self.node_embeddings = {}  # 重置

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)

            for node_info in nodes_data:
                node_id = node_info["node_id"]
                label = node_info.get("label", "unknown")
                mask = node_info.get("mask", "unknown")
                neighbors = node_info.get("neighbors", [])

                # 1) 计算文本向量
                text_content = node_info.get("text", "")
                embedding = self._compute_text_embedding(text_content)
                self.node_embeddings[node_id] = embedding

                # 2) 添加节点及其属性
                self.G.add_node(node_id, label=label, mask=mask)

                # 3) 添加无向边
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
        简易示例：将文本转换为随机向量。
        生产环境中请替换为真正的文本embedding逻辑。
        """
        rng = np.random.RandomState(len(text) + 2023)  # 做个简单区分
        # 这里随意设成128维
        return rng.rand(128)

    def _run_louvain_partition(self, gamma: float = 0.5):
        """
        自定义的 Louvain 社区检测，将语义相似度纳入到边权中。
        gamma 用于平衡结构信息(1) 和语义信息 d_sem(x_i, x_j)。
        """
        main_logger = logging.getLogger("main_logger")

        if self.G is None or self.G.number_of_nodes() == 0:
            self.partition = {}
            main_logger.warning("[GraphPerceptionAgent] Graph is empty, skip community detection.")
            return

        # 计算余弦相似度的内部函数
        def cosine_sim(vec1, vec2):
            if vec1 is None or vec2 is None:
                return 0.0
            dot_v = float(np.dot(vec1, vec2))
            norm1 = float(np.linalg.norm(vec1))
            norm2 = float(np.linalg.norm(vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_v / (norm1 * norm2)

        # 先清除已存在的 weight 属性
        for (u, v) in self.G.edges():
            if "weight" in self.G[u][v]:
                del self.G[u][v]["weight"]

        # 基于结构 (A_uv=1) + 语义相似度 来构建新的边权
        for (u, v) in self.G.edges():
            structural_part = 1.0
            emb_u = self.node_embeddings.get(u, None)
            emb_v = self.node_embeddings.get(v, None)
            semantic_part = cosine_sim(emb_u, emb_v)

            # 混合
            w_uv = gamma * structural_part + (1.0 - gamma) * semantic_part
            self.G[u][v]["weight"] = w_uv

        # 调用 python-louvain 的 best_partition，指定 weight='weight'。
        try:
            self.partition = community_louvain.best_partition(self.G, weight='weight', resolution=1.0)
            num_communities = max(self.partition.values()) + 1 if self.partition else 0
            main_logger.info(f"[GraphPerceptionAgent] (Semantic) Louvain found {num_communities} communities with gamma={gamma}.")
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error in custom semantic Louvain: {e}")
            self.partition = {}

    def _sort_communities_by_size(self):
        """
        根据每个社区的节点数降序排序社区 ID，并存入 self.sorted_communities。
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
        通过LLM选择一个较小的社区，计算PPR分数，并采样PPR分数最高的前10%节点。
        用于semantic增强模式。
        """
        main_logger = logging.getLogger("main_logger")

        if not self.partition:
            main_logger.info("[GraphPerceptionAgent] No partition available, so no sampling.")
            return []

        # 1) 收集社区分布信息
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

        # 2) 使用LLM选择一个较小的社区作为起点
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
            # 默认选择排序后后半部分的一个社区（避免选最小的）
            mid_point = len(self.sorted_communities) // 2
            selected_community_id = self.sorted_communities[mid_point] if self.sorted_communities else 0
        else:
            selected_community_id = int(community_selection["selected_community_id"])

        main_logger.info(f"[GraphPerceptionAgent] Selected community ID: {selected_community_id}")

        # 3) 计算所选社区的PPR分数
        self.ppr_scores = self._calculate_ppr_scores(selected_community_id)

        if not self.ppr_scores:
            main_logger.warning("[GraphPerceptionAgent] No PPR scores calculated. Falling back to random sampling.")
            # 如果PPR计算失败，从所有节点中随机抽取一些
            sample_size = min(self.sample_size, len(self.G.nodes))
            return random.sample(list(self.G.nodes), sample_size)

        # 4) 根据PPR分数排序节点
        sorted_nodes = sorted(self.ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # 5) 从PPR分数最高的前10%节点中随机采样
        top_percent = self.top_percent  # 前10%
        top_k = max(1, int(len(sorted_nodes) * top_percent))
        top_nodes = [node for node, _ in sorted_nodes[:top_k]]

        # 采样数量，不超过30个节点
        sample_size = min(self.sample_size, len(top_nodes))
        final_samples = random.sample(top_nodes, sample_size)

        main_logger.info(f"[GraphPerceptionAgent] Selected {sample_size} nodes from top {top_k} by PPR score")

        # 6) 如果需要，可视化采样结果
        if visualize and final_samples:
            self._visualize_sampling(final_samples)

        return final_samples

    def decide_label_sampling(self, target_label: int, label_target_size: int, visualize: bool = False) -> list:
        """
        当用户选择 'topological' 模式时，用于决定需要采样哪些节点做后续增强。
        这里同样使用PPR分数来选择与目标标签相关性高的节点。
        """
        main_logger = logging.getLogger("main_logger")
        if not target_label or label_target_size <= 0:
            main_logger.warning("[GraphPerceptionAgent] topological enhancement but invalid target_label or label_target_size.")
            return []

        # 找出所有符合该label的节点
        label_nodes = [n for n, attr in self.G.nodes(data=True)
                       if attr.get('label') == target_label and attr.get('mask') == "Train"]
        current_label_count = len(label_nodes)

        main_logger.info(f"[GraphPerceptionAgent] Found {current_label_count} nodes with label={target_label}.")

        # 如果已经达成目标，则不需要采样
        if current_label_count >= label_target_size:
            main_logger.info("[GraphPerceptionAgent] We already have enough nodes for that label.")
            return []

        # 如果目标标签的节点太少，计算PPR可能不稳定，则直接返回所有这些节点
        if len(label_nodes) <= 5:
            main_logger.info(f"[GraphPerceptionAgent] Too few nodes ({len(label_nodes)}) with label={target_label}, using all of them.")
            if visualize and label_nodes:
                self._visualize_sampling(label_nodes)
            return label_nodes

        # 为目标标签节点计算PPR分数
        temp_ppr_scores = {}
        for start_node in label_nodes:
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0
                ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)

                # 只关注与同标签节点的PPR分数
                for node in label_nodes:
                    temp_ppr_scores.setdefault(node, []).append(ppr[node])
            except:
                continue

        # 计算平均PPR分数
        avg_ppr_scores = {}
        for node, scores in temp_ppr_scores.items():
            if scores:
                avg_ppr_scores[node] = sum(scores) / len(scores)

        # 根据PPR分数排序节点
        if avg_ppr_scores:
            sorted_nodes = sorted(avg_ppr_scores.items(), key=lambda x: x[1], reverse=True)
            # 取前10%的节点
            top_percent = 0.1
            top_k = max(1, int(len(sorted_nodes) * top_percent))
            top_nodes = [node for node, _ in sorted_nodes[:top_k]]

            # 采样数量，不超过20个节点
            sample_size = min(20, len(top_nodes))
            sampled_nodes = random.sample(top_nodes, sample_size)
        else:
            # 如果PPR计算失败，随机采样
            sample_size = min(20, len(label_nodes))
            sampled_nodes = random.sample(label_nodes, sample_size)

        main_logger.info(f"[GraphPerceptionAgent] For label={target_label}, we sample {len(sampled_nodes)} nodes for enhancement.")

        if visualize and sampled_nodes:
            self._visualize_sampling(sampled_nodes)

        return sampled_nodes

    def _visualize_sampling(self, sampled_nodes):
        """
        可视化采样的节点，包括PPR分数信息。
        """
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 10))

        # 绘制所有节点（灰色）
        nx.draw_networkx_nodes(self.G, pos, node_color="lightgray", node_size=50, alpha=0.6)

        # 绘制采样节点（红色）
        nx.draw_networkx_nodes(self.G, pos, nodelist=sampled_nodes, node_color="red", node_size=100, alpha=0.9)

        # 如果有PPR分数，用颜色深浅表示PPR分数高低
        if self.ppr_scores:
            ppr_nodes = set(self.ppr_scores.keys()) - set(sampled_nodes)
            if ppr_nodes:
                ppr_values = [self.ppr_scores[node] for node in ppr_nodes]
                nx.draw_networkx_nodes(self.G, pos, nodelist=list(ppr_nodes),
                                       node_color=ppr_values, cmap=plt.cm.Blues,
                                       node_size=70, alpha=0.8)

                # 添加颜色条，显示PPR分数范围
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
                sm.set_array([min(ppr_values), max(ppr_values)])
                plt.colorbar(sm, label="PPR Score")

        # 绘制边
        nx.draw_networkx_edges(self.G, pos, width=0.5, alpha=0.3)

        # 为采样节点添加标签
        node_labels = {node: node for node in sampled_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8)

        plt.title("Sampled Nodes with PPR Scores Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("ppr_sampling_visualization.png", dpi=300)
        plt.show()

    def _calculate_ppr_scores(self, community_id):
        """
        计算从指定社区中的每个节点出发的个性化PageRank分数，并取平均值
        """
        main_logger = logging.getLogger("main_logger")

        # 获取指定社区中的所有节点
        community_nodes = [node for node, comm in self.partition.items() if comm == community_id]

        if not community_nodes:
            main_logger.warning(f"[GraphPerceptionAgent] No nodes found in community {community_id}")
            return {}

        main_logger.info(f"[GraphPerceptionAgent] Calculating PPR scores for {len(community_nodes)} nodes in community {community_id}")

        all_ppr_scores = {}
        for start_node in community_nodes:
            try:
                personalization = {node: 0.0 for node in self.G.nodes()}
                personalization[start_node] = 1.0
                ppr = nx.pagerank(self.G, alpha=0.85, personalization=personalization)

                # 累加每个节点的PPR分数
                for node, score in ppr.items():
                    all_ppr_scores.setdefault(node, []).append(score)
            except:
                main_logger.warning(f"[GraphPerceptionAgent] Failed to calculate PPR for node {start_node}")
                continue

        # 计算每个节点的平均PPR分数
        avg_ppr_scores = {}
        for node, scores in all_ppr_scores.items():
            if scores:  # 确保有分数才计算平均值
                avg_ppr_scores[node] = sum(scores) / len(scores)

        main_logger.info(f"[GraphPerceptionAgent] Calculated average PPR scores for {len(avg_ppr_scores)} nodes")
        return avg_ppr_scores

    def _call_generation(self, prompt: str) -> str:
        """
        调用LLM生成文本，增加错误处理和使用greedy decoding
        """
        main_logger = logging.getLogger("main_logger")
        try:
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,  # 改为False或自行调参
                temperature=0.7,
                top_p=1
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            main_logger.error(f"[GraphPerceptionAgent] Error during text generation: {e}")
            return "Error generating content. Please check the logs for details."

    def _extract_json(self, text: str) -> dict:
        """
        从LLM生成的文本中提取JSON数据
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

        # 再次尝试直接找 "selected_community_id"
        comm_id_pattern = r'"selected_community_id"\s*:\s*(\d+)'
        match = re.search(comm_id_pattern, text)
        if match:
            return {"selected_community_id": match.group(1)}

        return {}
