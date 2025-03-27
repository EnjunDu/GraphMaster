​	



![image-20250106175237807](../../../Users/JackDu/AppData/Roaming/Typora/typora-user-images/image-20250106175237807.png)

## LLM agent

* 统一采用**DeepSeek-R1 32B**，通过**Prompt模板** + **参数高效微调**（如LoRA / QLoRA / 4-bit 等）来完成所有Agent需要的自然语言处理、图文生成、评估反馈等功能。

![image-20250106155502553](../../../Users/JackDu/AppData/Roaming/Typora/typora-user-images/image-20250106155502553.png)

### Agent initial

* **Manager Agent：**
  * **Graph Perception Agent**：
    * 图数据解析和理解
    * 大图采样和特征提取
  * **Graph Enhancement Agent**
    * 统一处理文本和结构的生成
    * 确保生成的一致性和质量
    * 包含两个子模块：文本生成模块和结构生成模块
  * **Evaluation Agent**
    * 评估生成质量
    * 提供反馈

### 管理者Agent(Manager Agent)

* 为各个Agent分配资源
* 将大任务分解为子任务

#### 系统初始化与配置

* 初始化所有Agent并分配角色
* 设置全局参数(采样大小、质量阈值等)
* 建立Agent间的通信机制

#### 任务协调与调度

```python
def task_coordination:
    # 阶段1: 图感知与采样
    - 触发Graph Perception Agent进行采样
    - 等待并收集图结构和语义分析结果
    
    # 阶段2: 增强生成
    - 向Graph Enhancement Agent下发生成任务
    - 控制生成进度和规模
    - 监控生成过程的质量
    
    # 阶段3: 评估和反馈
    - 触发Evaluation Agent进行评估
    - 收集评估结果
    - 决定是否需要调整或继续迭代
```

#### 反馈整合机制

```python
def feedback_integration:
    # 收集反馈
    - 整合来自Evaluation Agent的质量评估
    - 分析生成效果
    
    # 策略调整
    - 根据反馈调整生成参数
    - 更新质量控制阈值
    - 修改采样策略
    
    # 终止判断
    - 评估是否达到目标
    - 决定是否需要额外迭代
```



### 感知Agent (Graph Perception Agent)

#### 采样

##### 初始采样

* 设定初始子图大小
* 使用随机游走/其他采样策略获取初始子图

##### 代表性采样

* 使用聚类方法对子图进行聚类
* 选择聚类中心为代表性样本
* 确保覆盖图的主要特征（强特征和弱特征都需覆盖）
* 实现具有代表性语义的子图采样

##### 感知环境

* 分析图结构特征
  * 节点的度，边的分布和权重等
* 提取文本语义信息
  * 分类标签、数值特征、语义特征等
* 生成环境状态报告，输入给其余**Agent**.

### 图增强Agent (Graph Enhancement Agent)

* 接收来自Graph Perception Agent的信息，了解图结构和语义信息

#### 节点文本生成/重构

##### 基于上下文的生成

* 考虑邻居节点的文本信息
* 保持语义连贯性
* 使用prompt引导生成

##### 属性增强

* 保持原始语义
* 增加多样性
* 确保合理性

#### 边生成/节点生成

* 根据节点语义关系
* 考虑图的结构特性
* 符合领域知识

```python
class GraphEnhancementAgent:
    def __init__(self):
        # Memory
        self.short_term_memory = {
            'current_subgraph': None,  # 当前处理的子图
            'generation_context': {},   # 生成上下文
            'neighbor_info': {}         # 邻居节点信息
        }
        self.long_term_memory = {
            'text_patterns': [],        # 成功的文本模式
            'edge_patterns': []        # 成功的边连接模式
        }
        
    def text_generation(self):
        """文本生成模块"""
        def context_analysis(node_id):
            # 分析节点上下文
            neighbors = self.get_neighbor_nodes(node_id)
            neighbor_texts = [self.get_node_text(n) for n in neighbors]
            return self.analyze_semantic_context(neighbor_texts)
            
        def generate_prompt(context):
            # 生成文本生成的prompt
            return f"""
            Based on the following context nodes: {context['neighbors']},
            Generate a new node text that:
            1. Maintains semantic coherence with neighbors
            2. Preserves original domain knowledge
            3. Introduces reasonable diversity
            Current focus: {context['focus']}
            Constraints: {context['constraints']}
            """
            
        def verify_generation(generated_text, context):
            # 验证生成的文本
            coherence_score = self.compute_semantic_coherence(generated_text, context)
            diversity_score = self.compute_diversity(generated_text, context)
            return coherence_score > 0.7 and diversity_score > 0.3
    
    def structure_enhancement(self):
        """结构增强模块"""
        def node_generation(subgraph):
            # 节点生成决策
            def decide_new_node_position():
                # 决定新节点的位置
                community_structure = self.analyze_communities(subgraph)
                weak_connections = self.find_weak_connections(community_structure)
                return self.select_optimal_position(weak_connections)
            
            def create_node_connections(new_node_id):
                # 创建新节点的连接
                potential_neighbors = self.find_potential_neighbors(new_node_id)
                connection_scores = self.evaluate_connections(new_node_id, potential_neighbors)
                return [(n, s) for n, s in connection_scores if s > 0.5]
                
        def edge_enhancement(subgraph):
            # 边增强决策
            def identify_missing_edges():
                # 识别潜在的缺失边
                node_pairs = self.get_potential_connections(subgraph)
                return self.rank_connection_probability(node_pairs)
            
            def validate_new_edges(candidate_edges):
                # 验证新边的合理性
                structural_scores = self.check_structural_validity(candidate_edges)
                semantic_scores = self.check_semantic_validity(candidate_edges)
                return [(e, s1*0.6 + s2*0.4) for e, s1, s2 in zip(candidate_edges, structural_scores, semantic_scores)]
    
    def quality_control(self):
        """质量控制模块"""
        def density_control(subgraph):
            # 控制图的密度
            current_density = self.calculate_density(subgraph)
            target_density = self.get_target_density()
            return self.adjust_density(current_density, target_density)
            
        def similarity_check(new_element, existing_elements):
            # 检查相似度
            similarities = self.compute_similarities(new_element, existing_elements)
            return all(s < self.thresholds['similarity'] for s in similarities)
            
        def structural_balance(subgraph):
            # 保持结构平衡
            metrics = {
                'degree_distribution': self.check_degree_distribution(subgraph),
                'clustering_coefficient': self.check_clustering(subgraph),
                'path_length': self.check_path_length(subgraph)
            }
            return self.evaluate_balance(metrics)
    
    def iteration_strategy(self):
        """迭代策略实现"""
        def single_iteration():
            # 单次迭代流程
            def generate_batch():
                # 生成新的批次
                node_quota = self.calculate_node_quota()  # 5% of total nodes
                edge_quota = self.calculate_edge_quota()  # based on density target
                return self.generate_elements(node_quota, edge_quota)
            
            def apply_changes(batch):
                # 应用更改
                success_rate = self.apply_batch_changes(batch)
                self.update_metrics(success_rate)
                return success_rate > self.thresholds['success']
                
            def update_strategy():
                # 更新生成策略
                self.adjust_thresholds()
                self.update_generation_parameters()
                
        def convergence_check():
            # 检查是否收敛
            quality_metrics = self.evaluate_current_quality()
            improvement_rate = self.calculate_improvement_rate()
            return quality_metrics > self.thresholds['quality'] and improvement_rate < self.thresholds['improvement']
```

#### 迭代策略

* 每轮生成节点数限制（比如不超过总节点数的5%）
* 边密度控制
* 质量阈值设定（使用相似度算法等，新生成的节点/属性应与其他节点相似度低于阈值$\theta$.

### 评估Agent (Evaluation Agent)

#### 质量评估

* 结构评估
  * 连通性分析
  * 度分布评估
  * 聚类系数计算
* 语义评估
  * ROUGE-L评分
  * 节点间语义关联度

#### 质量控制机制

* 阈值控制——设置阈值$\theta$评估生成内容质量
* 反馈生成——删除低质量生成内容，并且生成反馈
* 评估结果存储——利用Memory，存储最重要的评估信息，以方便迭代

## 微调策略

* 考虑到成本与内存&推理时间，我们选取**一个统一的基础LLM + 针对性微调/Prompt Engineering** 
* 使用参数高效微调(如LoRA)

![image-20250106180407801](../../../Users/JackDu/AppData/Roaming/Typora/typora-user-images/image-20250106180407801.png)

### Manager Agent

* **无需进行**微调，主要进行任务协调和决策，可通过prompt templates实现

```python
MANAGER_PROMPT_TEMPLATE = """
Role: You are a coordinator managing graph data synthesis tasks.
Context: {current_state}
Task History: {task_history}
Available Resources: {resources}

Please:
1. Analyze the current state
2. Decide next steps
3. Assign tasks to appropriate agents
4. Monitor progress and provide feedback

Constraints:
- Consider resource limitations
- Maintain task dependencies
- Ensure quality control
"""
```

### Graph Perception Agent

* 理解图结构和特征
* 专注于图分析和采样任务
* 需进行**轻量级**微调

```python
def perception_agent_finetuning:
    # 1. 图结构理解任务
    tasks = [
        "graph_structure_analysis",
        "node_feature_extraction",
        "subgraph_sampling"
    ]
    
    # 2. 训练数据准备
    training_data = {
        "graph_examples": [...],  # 各种规模和类型的图
        "sampling_results": [...], # 标准采样结果
        "feature_annotations": [...] # 特征标注
    }
    
    # 3. 特定任务微调
    training_config = {
        "learning_rate": 1e-5,
        "epochs": 5,
        "batch_size": 16
    }
```

### Graph Enhancement Agent

* 需要深入理解文本和图结构
* 生成任务要求高质量输出
* 需要进行**详细、高质量**的微调

```python
def enhancement_agent_finetuning:
    # 1. 文本生成微调
    text_generation_tasks = {
        "context_aware_generation": {...},
        "semantic_preservation": {...},
        "style_consistency": {...}
    }
    
    # 2. 结构生成微调
    structure_enhancement_tasks = {
        "node_generation": {...},
        "edge_prediction": {...},
        "graph_completion": {...}
    }
    
    # 3. 分阶段微调策略
    finetuning_stages = [
        {
            "stage": "text_understanding",
            "data": text_corpora,
            "epochs": 3
        },
        {
            "stage": "Synthesis",
            "data": graph_datasets,
            "epochs": 3
        },
        {
            "stage": "joint_training",
            "data": combined_datasets,
            "epochs": 2
        }
    ]
```

### Evaluation Agent

* 需要掌握评估标准
* 能够生成有效反馈
* 进行**轻量级**微调

```python
def evaluation_agent_finetuning:
    # 1. 评估标准学习
    evaluation_criteria = {
        "structural_metrics": [...],
        "semantic_metrics": [...],
        "quality_thresholds": [...]
    }
    
    # 2. 反馈生成训练
    feedback_training = {
        "positive_examples": [...],
        "negative_examples": [...],
        "improvement_suggestions": [...]
    }
```



# baseline

## Evaluation indicators：

* accuracy_score
* precision_score
* recall_score
* f1_score

## Class imbalance

* 数据集中是否存在某些类别过多或过少，增强后的数据对不平衡问题是否有所改善？

  * 人为制造少样本数据
  * 数据增强后运用Evaluation indicators进行评估

  - 如果增强对少数类起到了改善作用，那么Macro-F1一般会提升。

## 拓扑 imbalance

* 图在拓扑结构上是否存在某些异常或不平衡的部分？如：

  * 度分布是否很极端(有些节点度很高、有些节点度很低)；

  * 大小社区不均衡(某些子图特别大，某些子图特别小)；

  * 不同的模态/簇之间结构差距过大等。

* 验证方法

  * 度分布图(如直方图/Log-Log分布)
  * 社区检测算法(Louvain, Girvan-Newman等)对图进行划分；

* 拓扑不平衡中的同配与异配

  * 同配指的是**相似的节点更倾向于相互连接**。
  * 异配则是**不同的节点也具有较高的连接概率**。
  * 拓扑不平衡不仅体现在**类别分布不均**，还体现在**同配与异配的连接模式**上。
    * **同配倾向过强或过弱**：同类节点过度聚集或分散，影响信息传播和模型学习。
    * **异配连接比例异常**：跨类别连接过多或过少，可能导致类别间信息交互不充分或噪声过多。
    * **社区结构不均衡**：某些社区规模过大，另一些过小，导致模型在不同社区中的表现不一致。

* 验证方法：

  * 同配指数（Homophily Index）：
    $$
    H=\frac{同类边数}{总边数}
    $$

  * 模块度
    $$
    Q=\frac{1}{2m}\sum_{i,j}[A_{i,j}-\frac{k_ik_j}{2m}]\delta(c_i,c_j)
    $$

    * $A_{i,j}$：节点$i$和节点$j$之间的实际连接
    * $k_i$,$k_j$：节点$i$和节点$j$的度
    * $m$:图中总边数
    * $\delta(c_i,c_j)$当节点$i$和节点$j$属于同一社区时为1，否则为0

  

## few-shot

* 当标注样本极少(例如每个类只有少量标注)，模型能否仍然获得好的性能？增强或合成数据对“小样本场景”有何帮助？
  * 从原始数据中**仅保留极少数有标签的节点/子图**，模拟极端的小样本环境
* 依旧以Evaluation indicators为评判标准
* 做多次随机采样(多次分不同few-shot子集)求均值和方差，看模型是否有较大波动。

## sparsity

* 图是否过于稀疏（节点多，边少），增强过程是否加剧或减轻了稀疏问题？
* 验证方法：
  * 边-节点比 (Edge-to-Node Ratio)——直接统计边数/节点数，或平均节点度 `2 * edges / nodes`；
  * 子图内部的稀疏度(如`subEdges / subNodes`)；

## noise

* 数据中存在随机噪声或系统性噪声时，增强方法是否对噪声具有鲁棒性，或者能否去噪？
  * **结构噪声**：随机添加/删除一部分边，或错误的节点连接；
  * **特征噪声**：节点特征或文本属性中存在随机扰动/缺失，或错误标签；
  * **标签噪声**：部分节点/图被标错标签。
  * 人为进行噪声注入
* 依旧以Evaluation indicators为评判标准
