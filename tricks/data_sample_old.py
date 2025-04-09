# data_sample.py
import networkx as nx
import json
import random
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

# =========================
# 1. 读取边数据，构建图
# =========================
edges_file = "edges.txt"  # 边文件路径
edges = []

# 读取边文件，解析节点对
with open(edges_file, "r") as file:
    for line in file:
        node1, node2 = map(int, line.strip().split())  # 解析每一行的两个节点
        edges.append((node1, node2))

# 构建 NetworkX 图
G = nx.Graph()
G.add_edges_from(edges)

# =========================
# 2. 运行 Louvain 社区检测
# =========================
partition = community_louvain.best_partition(G)  # 计算 Louvain 社区
num_communities = max(partition.values()) + 1  # 计算社区数量
print(f"Detected {num_communities} communities")

# 统计每个社区的节点
community_nodes = {}
for node, comm in partition.items():
    community_nodes.setdefault(comm, []).append(node)

# =========================
# 3. 社区按照节点数量从大到小排序
# =========================
sorted_communities = sorted(
    community_nodes.keys(),
    key=lambda c: len(community_nodes[c]),
    reverse=True
)

print("Communities sorted by size (descending):")
for idx, comm in enumerate(sorted_communities):
    print(f"Rank {idx}: Community {comm} has {len(community_nodes[comm])} nodes")

# =========================
# 4. 读取用户输入
# =========================
user_input = input("\nEnter start_community_index, end_community_index, sample_size_k (separated by space): ")
start_community_index, end_community_index, sample_size_k = map(int, user_input.split())

# 安全检查：如果 end_community_index 超过可用范围，则截断到最后
end_community_index = min(end_community_index, len(sorted_communities) - 1)

# =========================
# 5. 合并区间社区的节点
# =========================
selected_nodes_all = []
for idx in range(start_community_index, end_community_index + 1):
    comm_id = sorted_communities[idx]
    selected_nodes_all.extend(community_nodes[comm_id])

total_nodes_count = len(selected_nodes_all)
print(f"Selected communities from rank {start_community_index} to {end_community_index}.")
print(f"Total nodes in these communities: {total_nodes_count}")

# =========================
# 6. 采样指定数量 k 个节点
# =========================
sample_k = min(sample_size_k, total_nodes_count)
sampled_nodes = random.sample(selected_nodes_all, sample_k)
print(f"Randomly sampled {len(sampled_nodes)} nodes from the selected communities.")

# =========================
# 7. 输出采样结果（排序后）
# =========================
output_path = "selected_community_sampled_nodes.json"
sampled_nodes_sorted = sorted(sampled_nodes)  # 对采样节点进行从小到大的排序
output_str = ", ".join(map(str, sampled_nodes_sorted))
print(output_str)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_str)

print(f"Sampled nodes saved to {output_path}")

# =========================
# 8. 可视化（可选）
# =========================
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)  # 计算布局
# 不同社区使用不同颜色
colors = [partition[n] for n in G.nodes()]
nx.draw(G, pos, node_color=colors, node_size=30, with_labels=False, edge_color="gray", alpha=0.3)

# 高亮显示采样节点
highlight_nodes = set(sampled_nodes_sorted)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(highlight_nodes),
    node_color="red",
    node_size=50
)

plt.title(f"Louvain Community Detection - Sampled {sample_k} nodes from communities rank {start_community_index}~{end_community_index}")
plt.show()
