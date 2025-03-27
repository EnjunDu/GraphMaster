import json
import random

dataset = "cora"
# 读取cora.json
with open(f'../data/{dataset}.json', 'r') as f:
    data = json.load(f)

# 统计每个label的节点个数
label_counts = {}
for node in data:
    label = node['label']
    if node['mask'] == 'Train':
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

# 输出每个label的节点个数
print("Train Label counts:", label_counts)

# 获取用户输入的x和y
x = int(input("Enter label value (x): "))
y = int(input("Enter number of nodes to keep (y): "))

# 先将所有mask为'train'且label为x的节点收集到一个列表中
train_x_nodes = [node for node in data if node['label'] == x and node['mask'] == 'Train']

# 确保train_x_nodes列表的长度至少为y
if len(train_x_nodes) < y:
    print(f"Warning: There are fewer than {y} nodes with label {x} and mask 'train'. All {len(train_x_nodes)} nodes will be kept.")
    selected_nodes = train_x_nodes  # 如果不足y个节点，则保留所有该label和mask条件的节点
else:
    # 随机选择y个节点
    selected_nodes = random.sample(train_x_nodes, y)

# 创建一个删除节点的集合
deleted_nodes = set(node['node_id'] for node in train_x_nodes if node not in selected_nodes)

# 创建新数据列表，保留随机选择的label为x且mask为'train'的节点，其他节点不变
new_data = []
for node in data:
    # 保留所有的节点，mask非'train'的节点不做任何更改
    if node['label'] != x or (node['mask'] != 'Train' or node in selected_nodes):
        new_data.append(node)

# 遍历所有节点的neighbors，删除已经删除的节点
for node in new_data:
    if 'neighbors' in node:
        # 过滤掉已经删除的节点
        node['neighbors'] = [neighbor for neighbor in node['neighbors'] if neighbor not in deleted_nodes]

# 重新调整node_id，使其从0开始连续
id_mapping = {}
new_node_id = 0

# 对new_data中的所有节点进行重排
for node in new_data:
    id_mapping[node['node_id']] = new_node_id
    node['node_id'] = new_node_id
    new_node_id += 1

# 更新所有节点的neighbors，使用新的node_id
for node in new_data:
    if 'neighbors' in node:
        # 使用id_mapping更新neighbors中的node_id
        updated_neighbors = []
        for neighbor in node['neighbors']:
            if neighbor in id_mapping:  # 只更新存在id_mapping中的邻居
                updated_neighbors.append(id_mapping[neighbor])
        node['neighbors'] = updated_neighbors

# 将修改后的数据保存为cora_label:{x}_{y}.json
output_filename = f"../data/{dataset}_label_{x}_{y}.json"
with open(output_filename, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Modified data saved to {output_filename}")
