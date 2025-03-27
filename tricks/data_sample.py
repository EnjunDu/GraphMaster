# data_sample.py

import networkx as nx
import json
import random
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt


def load_graph_from_edges_file(edges_file: str) -> nx.Graph:
    """
    从给定的边列表文件中读取边，并构建无向图。
    :param edges_file: 边文件路径，每行两个整数，表示 (node1, node2)
    :return: 构建好的 NetworkX 无向图
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
    对图 G 进行 Louvain 社区检测，返回社区划分字典和总社区数量。
    :param G: 输入的 NetworkX 图
    :return:
        partition: dict，key 是 node，value 是社区编号
        num_communities: int，社区总数量
    """
    partition = community_louvain.best_partition(G)
    num_communities = max(partition.values()) + 1 if partition else 0
    return partition, num_communities


def sort_communities_by_size(partition: dict) -> list:
    """
    根据社区大小从大到小返回社区编号的排序列表。
    :param partition: Louvain 社区分割结果
    :return: list, 按大小降序排序的社区编号
    """
    # 收集社区 -> 节点列表
    community_nodes = {}
    for node, comm_id in partition.items():
        community_nodes.setdefault(comm_id, []).append(node)

    # 根据社区大小进行排序
    sorted_communities = sorted(
        community_nodes.keys(),
        key=lambda c: len(community_nodes[c]),
        reverse=True
    )
    return sorted_communities


def visualize_communities(G: nx.Graph, partition: dict, sampled_nodes=None, title="Louvain Community Detection"):
    """
    可视化网络图及社区划分结果，支持高亮显示采样节点。
    :param G: NetworkX 图
    :param partition: Louvain 社区分割结果
    :param sampled_nodes: 要高亮显示的节点列表
    :param title: 图像标题
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # 计算布局(随机, 也可换成其他)

    # 使用社区编号作为颜色
    colors = [partition[n] for n in G.nodes()]
    nx.draw(
        G, pos,
        node_color=colors,
        node_size=30,
        with_labels=False,
        edge_color="gray",
        alpha=0.3
    )

    # 如果有采样节点，则将其在图中高亮
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
    edges_file = "edges.txt"  # 默认边文件路径
    print(f"[data_sample.py] Loading edges from {edges_file} ...")
    G = load_graph_from_edges_file(edges_file)
    print(f"[data_sample.py] Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("[data_sample.py] Running Louvain community detection...")
    partition, num_communities = run_louvain_community_detection(G)
    print(f"[data_sample.py] Detected {num_communities} communities.")

    # 按社区大小排序
    sorted_communities = sort_communities_by_size(partition)

    # 打印排序结果
    print("Communities sorted by size (descending):")
    for idx, comm in enumerate(sorted_communities):
        # 统计该社区节点个数
        node_count = sum(1 for node, c in partition.items() if c == comm)
        print(f"Rank {idx}: Community {comm} has {node_count} nodes")

    # 读取用户输入
    print("\nEnter 'start_community_index end_community_index sample_size_k', e.g. '0 1 50'")
    user_input = input(">>> ")
    try:
        start_community_index, end_community_index, sample_size_k = map(int, user_input.split())
    except:
        print("[Error] Invalid input. Please input 3 integers separated by space.")
        exit(1)

    # 防御性检查
    if start_community_index < 0 or start_community_index >= len(sorted_communities):
        print("[Warning] start_community_index out of range. Force set to 0.")
        start_community_index = 0
    if end_community_index < start_community_index:
        print("[Warning] end_community_index < start_community_index. Force set to the same as start.")
        end_community_index = start_community_index
    if end_community_index >= len(sorted_communities):
        print(f"[Warning] end_community_index out of range. Force set to {len(sorted_communities) - 1}.")
        end_community_index = len(sorted_communities) - 1

    # 收集多个社区节点
    selected_nodes_all = []
    for idx in range(start_community_index, end_community_index + 1):
        comm_id = sorted_communities[idx]
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        selected_nodes_all.extend(comm_nodes)

    total_nodes_count = len(selected_nodes_all)
    print(f"[data_sample.py] Selected communities from rank {start_community_index} to {end_community_index}.")
    print(f"[data_sample.py] Total nodes in these communities: {total_nodes_count}")

    # 采样
    sample_k = min(sample_size_k, total_nodes_count)
    sampled_nodes = random.sample(selected_nodes_all, sample_k)
    sampled_nodes_sorted = sorted(sampled_nodes)
    print(f"[data_sample.py] Randomly sampled {len(sampled_nodes_sorted)} nodes. Below are the sorted IDs:\n")
    output_str = ", ".join(map(str, sampled_nodes_sorted))
    print(output_str)

    # 写入文件
    output_path = "selected_community_sampled_nodes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    print(f"[data_sample.py] Sampled nodes saved to {output_path}")

    # （可选）可视化结果
    visualize_communities(
        G, partition,
        sampled_nodes=sampled_nodes_sorted,
        title=f"Louvain - Sample {sample_k} from communities [{start_community_index}..{end_community_index}]"
    )
