# GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments



<p align="center">
<img src=tricks\GraphMaster.png alt="GraphMaster" width="40%" />
</p>


GraphMaster is a novel multi-agent system for **graph data enhancement**, built upon the **Retrieval-Augmented Generation (RAG)** paradigm and powered by **Large Language Models (LLMs)**. It is designed for few-shot or low-resource graph learning tasks, where both **semantic diversity** and **structural quality** are critical.

## 🚀 Key Features

- **Multi-Agent Architecture** simulating human-in-the-loop perception, enhancement, evaluation, and management.
- **RAG-based Iterative Enhancement** over graph data using LLMs.
- **Semantic & Topological Modes** for diversified and structure-aware node generation.
- **Auto-Adaptive Objective Weights** across semantic, structural, and label balance metrics.
- **Plug-and-Play LLMs**: Easily switch between Qwen, Deepseek, LLaMA, or any HF-supported model.
- **Data-Limited Datasets**: For more details, please refer [another README](./data/README.md).

## 🧠 Architecture

```
+--------------------+     +--------------------+     +------------------------+
|  Perception Agent  | --> | Enhancement Agent  | --> | Evaluation Agent       |
+--------------------+     +--------------------+     +------------------------+
          ^                                                  |
          |                                                  v
    +--------------------+                          +------------------+
    |   Manager Agent     |<------------------------|   Enhanced Graph |
    +--------------------+                          +------------------+
```

## 📂 Project Structure

```
\src
├── main.py                    # Entry point
├── manager_agent.py           # Agent that controls the full pipeline
├── perception_agent.py        # Builds graph, samples subgraphs, computes stats
├── enhancement_agent.py       # Generates new nodes (semantic/topological)
├── evaluation_agent.py        # Evaluates generated nodes and detects convergence
├── data/
│   └── cora.json              # Input graph (JSON format)
\data                          # data-limited datasets, and the corresponding generate data 
\log                           # logs while run the pipline
\old_codes                     # old version of GraphMaster
\tricks                        # Some preprocessing codes
\Vertification                 # GNN verification model, used for Bert&GNN to verify data effects
```



## 📦 Installation

```
conda create -n graphmaster python=3.11
conda activate graphmaster
pip install -r requirements.txt
```

> Requirements include `transformers`, `networkx`, `scikit-learn`, `community` (for Louvain), `matplotlib`
>
> The experiment is best run on either 8 A6000 GPUs with 48GB memory each or 4 A100 GPUs with 80GB memory each. However, based on our experiments, a single A100 GPU with 80GB memory can also run the experiment, albeit with a significant increase in runtime.

## 📄 Input Format

Each node is described in JSON:

```
{
  "node_id": "123",
  "label": 2,
  "text": "A novel GNN model is proposed...",
  "neighbors": ["45", "78"],
  "mask": "Train"
}
```

## 🧪 Running the Pipeline

```
cd src
python main.py \
  --data_file ./data/SubCora.json \
  --llm_model QwQ \
  --enhancement_mode semantic \
  --max_iterations 10 \
  --visualize_sampling
```

#### Supported `--llm_model`:

- `Qwen` → Qwen1.5-32B
- `Deepseek` → DeepSeek-R1-Distill-Qwen-32B
- `LLaMA` → Samantha 1.1 (LLaMA 33B)
- `QwQ` → Qwen/QwQ-32B (preview model)

> Custom models also supported by providing HF path.

## 📈 Outputs

- Enhanced graph stored in `cora_enhanced.json`
- Adaptive weights saved per iteration
- Visualizations:
  - `adaptive_weights_evolution.png`
  - `label_distribution_change.png`

## 🤖 Agent Highlights

### PerceptionAgent

- Graph construction (using NetworkX)
- Louvain community detection with semantic similarity
- PPR-based sampling from high-variance community

### EnhancementAgent

- Prompt-based LLM generation
- Supports both `semantic` and `topological` enhancements
- Edge construction via probabilistic model (sim + overlap + centrality)

### EvaluationAgent

- Computes composite quality score (0-10 scale)
- Adaptive threshold & early stopping
- Convergence analysis using quality gradients + LLM summary

### ManagerAgent

- Controls the full loop
- Auto-selects enhancement mode based on multi-objective utility
- Updates adaptive weights (λ₁, λ₂, λ₃)

## 📊 Citation-Style Motivation

> "GraphMaster simulates a human-guided editing process on attributed graphs by iteratively improving data with structured perception, controlled generation, and critical evaluation — powered by LLMs."



## 📘 License

MIT License
