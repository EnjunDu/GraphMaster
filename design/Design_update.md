### 理解数据

graph edit  llm 4 rgnn 

**让大语言模型（LLM）理解原始图数据**（adj.pkl、features.pkl、labels.pkl、nids.pkl），将这些文件转换为适合 LLM 输入的形式。

* 对于每个节点的描述如下：

  ```python
  node_i = {
      "node_id": 123, # 表示它是第几个，name
      "label": "Neural_Networks", # 标签
      "text": "Paper about neural networks applied to classification ...",
      "neighbors": [45, 78, 600]          
  }
  ```

* 分块输入：

  * 随机游走 (Random Walk)
  * **BFS/DFS**
  * 社区聚类

* 图增强：

  * **输入**：节点信息+prompt
  * **输出**：新的节点或新的边，以及简单的解释。

  ```python
  System: "You are a Graph Enhancement Agent."
  User: "Here is a subgraph with 5 nodes:
         Node0(label=NN,text='...'), Node1(label=SVM,text='...'), ...
         Task: Add 1 or 2 new nodes that link the subgraph to an external concept about 'Bayesian Methods'.
         Keep the new nodes' text consistent with the domain.
         Return the updated edges and node texts."
  ```

  



## 实验方法：

### sparsity

* 进行社区聚类，将稀疏节点（同时将稀疏节点的邻居节点、以及随机挑选与稀疏节点同label的节点）喂给大语言模型，使大语言模型生成新的边。
* 检测增强前后模型训练能力。

### few-shot——同（Class imbalance）

* 将某一个label删除至5、10、20个节点，然后反复投喂给大语言模型，直至生成3*n个标注节点。
* 检测增强前后模型训练能力。

### noise

* 随机删除一些边，并且对删除的边进行标记，判断大语言模型进行边增强后模型的训练效果
* 随机增加一些无关边（不同属一个类别），判断大语言模型进行边增强后的模型训练效果

### 拓扑 imbalance

* 判断大语言模型进行增强后，同属一个label的节点的连接情况



## 数据格式

```json
  {
    "node_id": 0,
    "label": 2,
    "text": "Title:... Abstract:... ",
    "neighbors": [
      8,
      14,
      258,
      435,
      544
    ],
    "mask": "Test"
  },
```

## Prompt

```
"""# Graph Data Augmentation Task

You are a Graph Data Enhancement Agent. You are tasked with performing data augmentation on graph data. The graph nodes are described by the following structure: {{node_id, label, text, neighbors, mask}}, where the `mask` field can be one of the following: **Train**, **Test**, or **Validation**.

## Relevant Environment State:
{environment_state_str}

## Your augmentation tasks are as follows:

1. **Text Modification**: You Can modify the text description of each node while preserving its semantic meaning exactly.

2. **Neighbor Modification**: You Cam modify or add/remove neighbors based on the global semantic relationships between nodes, ensuring that the overall connectivity properties of the graph are maintained.

3. **New Node Addition**: You Can add new nodes to enhance data diversity. if you choose to do this, each new node MUST follow the same format as existing nodes, that is: {{new_node_id, label, text, neighbors, mask}}. The mask for every new node MUST be **Train**. The total number of new nodes added MUST NOT exceed 5% of the total number of input nodes (if 5% is fractional, round down to the nearest integer). Newly added nodes MUST have unique `new_node_id` values, starting from **"new_node 1"** and incrementing sequentially.

4. **augmentation overall task**: The mission of the enhancement task is to increase the diversity and comprehensiveness of the data and make the connectivity between nodes with the same label higher. You can choose one or more of the three methods 1, 2, and 3 to complete the task. For some nodes that you think are already perfect, you can choose not to perform any operation.

IMPORTANT:
- You MUST strictly adhere to all of the above constraints.
- You MUST output the final enhanced graph data in **pure JSON format** with no additional explanations, annotations, or extra text.
- When you are ready to answer, you MUST first output exactly the following line:  
  **"here are the generated datasets:"**  
  followed immediately by the JSON data, and nothing else.

Here is the graph data that needs augmentation:
{data_json_str}

You may think about the task step by step internally, but your final output MUST exactly follow the instructions above.
"""
```

```
# Graph Data Evaluation Task
You are a Graph Data Evaluation Agent. You are tasked with evaluating the quality of the graph data that was generated. The graph nodes are described with the following structure: `{node_id, label, text, neighbors, mask}`, where the `mask` field can have one of three types: **Train**, **Test**, or **Validation**.

## Your evaluation tasks are as follows:

1. **Connectivity Analysis**: Evaluate the connectivity of the graph. Are there any newly added nodes which seem to be a isolated subgraphs or nodes?

2. **Degree Distribution**: Analyze the degree distribution of the graph. Is it reasonable? Are there any nodes with very high or very low degrees?

3. **Clustering Coefficient**: Calculate the clustering coefficient. Does the graph exhibit appropriate community structure?

4. **Semantic Evaluation**:
   - **Text Evaluation**: Evaluate whether the newly generated text maintains semantic coherence with the surrounding nodes. Use ROUGE-L score for this purpose.
   - **Semantic Consistency**: Ensure that the node text is consistent with its label and neighboring nodes' texts.

5. **Homophily Index**: Calculate the homophily index of the graph. Does the graph exhibit a reasonable homophily or heterophily pattern?

6. **Overall Quality**: Provide an overall evaluation of the graph's quality. Is the structure and semantics consistent with the original graph's properties?

## If you think a piece of data is unreasonable, or you think a neighbor should not exist, you can directly delete the data or neighbor. Please note that your operation should only be limited to newly added nodes or neighbors, and you need to try to avoid damaging existing nodes or neighbors.
Here is the original and generated data that needs evaluation:
original: {selected_data}
generated : {generated_data}
You can think about it step by step, but when you are ready to answer, you must first say "here are the generated datasets:" and then directly output the JSON format of the data after evaluating by you  after thinking. When you say "here are the generated datasets:", you can only output standard data without any other explanation.
""
```

