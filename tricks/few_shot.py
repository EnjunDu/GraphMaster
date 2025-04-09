import json
import random

dataset = "cora"
# Read cora.json
with open(f'../data/{dataset}.json', 'r') as f:
    data = json.load(f)

# Count the number of nodes for each label
label_counts = {}
for node in data:
    label = node['label']
    if node['mask'] == 'Train':
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

# Output the number of nodes for each label
print("Train Label counts:", label_counts)

# Get the x and y values ​​entered by the user
x = int(input("Enter label value (x): "))
y = int(input("Enter number of nodes to keep (y): "))

# First collect all nodes with mask 'train' and label x into a list
train_x_nodes = [node for node in data if node['label'] == x and node['mask'] == 'Train']

# Make sure the train_x_nodes list is at least as long as y
if len(train_x_nodes) < y:
    print(f"Warning: There are fewer than {y} nodes with label {x} and mask 'train'. All {len(train_x_nodes)} nodes will be kept.")
    selected_nodes = train_x_nodes  # If there are less than y nodes, keep all nodes with the label and mask conditions.
else:
    # Randomly select y nodes
    selected_nodes = random.sample(train_x_nodes, y)

# Create a collection of deleted nodes
deleted_nodes = set(node['node_id'] for node in train_x_nodes if node not in selected_nodes)

# Create a new data list, keep the randomly selected nodes with label x and mask 'train', and keep other nodes unchanged
new_data = []
for node in data:
    # Keep all nodes, and do not make any changes to nodes that are not 'train' in mask
    if node['label'] != x or (node['mask'] != 'Train' or node in selected_nodes):
        new_data.append(node)

# Traverse the neighbors of all nodes and delete the deleted nodes
for node in new_data:
    if 'neighbors' in node:
        # Filter out deleted nodes
        node['neighbors'] = [neighbor for neighbor in node['neighbors'] if neighbor not in deleted_nodes]

# Rearrange node_id to make it consecutive starting from 0
id_mapping = {}
new_node_id = 0

# Rearrange all nodes in new_data
for node in new_data:
    id_mapping[node['node_id']] = new_node_id
    node['node_id'] = new_node_id
    new_node_id += 1

# Update neighbors of all nodes, using the new node_id
for node in new_data:
    if 'neighbors' in node:
        # Use id_mapping to update node_id in neighbors
        updated_neighbors = []
        for neighbor in node['neighbors']:
            if neighbor in id_mapping:  # Only update neighbors that exist in id_mapping
                updated_neighbors.append(id_mapping[neighbor])
        node['neighbors'] = updated_neighbors

# Save the modified data as cora_label:{x}_{y}.json
output_filename = f"../data/{dataset}_label_{x}_{y}.json"
with open(output_filename, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Modified data saved to {output_filename}")
