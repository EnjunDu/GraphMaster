# enhancement_agent.py
import json
import time
import threading
import numpy as np
import random
from transformers import TextGenerationPipeline
import logging


class GraphEnhancementAgent:
    """
    Enhancement Agent - implements the generation component of the RAG paradigm:
    - Generates node attributes using a conditional autoregressive model
    - Models edge connections using a probability function
    - Supports both semantic and topological enhancement modes
    - Dynamically adjusts generation parameters based on mode
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: LLM text generation pipeline
        :param max_new_tokens: Maximum new tokens for generation
        """
        self.text_generation = text_generation_pipeline
        self.max_new_tokens = max_new_tokens

    def enhance_graph(self, data_json_str: str, environment_state_str: str = "",
                      mode: str = "semantic",
                      target_label: str = None,
                      label_target_size: int = 0
                      ) -> str:
        """
        Executes graph enhancement based on the RAG paradigm's generation component.
        
        For semantic mode, implements a conditional autoregressive model:
        P(x_s | K_t, R_t) = ∏_i P(x_s^i | x_s^<i, X_k, E_k, R_t)
        
        For topological mode, models edge connections with probability:
        P((v_s, v_i) ∈ E_c | K_t, R_t) = σ(θ₁·sim(x_s, x_i) + θ₂·|N(v_i) ∩ N_K(v_s)|/|N_K(v_s)| + θ₃·k_i/max_j k_j)
        
        :param data_json_str: Input graph data (JSON string)
        :param environment_state_str: Environment report
        :param mode: 'semantic' or 'topological'
        :param target_label: Target label for topological mode
        :param label_target_size: Target size for topological mode
        :return: Enhanced graph data (JSON string)
        """
        if mode == "topological" and target_label:
            # Generate prompt for topological enhancement
            prompt_for_enhance = self._create_topological_enhancement_prompt(
                data_json_str, environment_state_str, target_label, label_target_size)
        else:  # mode == "semantic"
            # Generate prompt for semantic enhancement
            prompt_for_enhance = self._create_semantic_enhancement_prompt(
                data_json_str, environment_state_str)

        main_logger = logging.getLogger("main_logger")
        main_logger.info(f"[GraphEnhancementAgent] LLM begin to enhance the given graph data with mode={mode}.")

        # Generate enhanced data using LLM
        raw_output = self._call_generation(prompt_for_enhance, self.max_new_tokens)
        splitted_flag = "here are the generated datasets:"
        enhanced_json_str = self._extract_after_flag(raw_output, splitted_flag)
        
        if not enhanced_json_str.strip():
            main_logger.warning("[GraphEnhancementAgent] Warning: LLM returned empty enhancement result.")
            return "[]"
        
        # Postprocess the enhanced data to ensure it matches our model requirements
        try:
            # Parse the enhanced data
            enhanced_data = json.loads(enhanced_json_str)
            
            # Apply mode-specific edge probability model
            enhanced_data = self._apply_edge_probability_model(enhanced_data, data_json_str, mode)
            
            # Convert back to string
            enhanced_json_str = json.dumps(enhanced_data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            main_logger.error(f"[GraphEnhancementAgent] Error parsing enhanced data: {e}")
            # Return original string if parsing fails
            enhanced_json_str = enhanced_json_str  # Keep as is
        
        return enhanced_json_str

    def _create_semantic_enhancement_prompt(self, data_json_str, environment_state_str):
        """
        Creates a prompt for semantic enhancement that implements the conditional
        autoregressive model for generating node attributes.
        """
        prompt = f"""# Graph Data Augmentation Task: Semantic Enhancement

You are a Graph Data Enhancement Agent. You are tasked with performing semantic enhancement on graph data. The graph nodes are described by the following structure: {{node_id, label, text, neighbors, mask}}, where the `mask` field can be one of the following: **Train**, **Test**, or **Validation**.

## Semantic enhancement focuses on:
- Improving data quality and diversity
- Enhancing the meaning and content of nodes
- Adding variations to existing semantic patterns
- Enriching the textual information while maintaining label consistency

## Relevant Environment State:
{environment_state_str}

## Your augmentation tasks are as follows:

1. **New Node Addition**: You MUST add new nodes to enhance data diversity. Each new node MUST follow the same format as existing nodes, that is: {{new_node_id, label, text, neighbors, mask}}. The mask for every new node MUST be **Train**. The total number of new nodes added MUST NOT exceed 10% of the total number of input nodes (if 10% is fractional, round down to the nearest integer). Newly added nodes MUST have unique `new_node_id` values, starting from **"new_node 1"** and incrementing sequentially.

2. **Semantic Enhancement Using Conditional Autoregressive Model**: Generate text for each new node token by token, where each token depends on:
   - Previously generated tokens in the sequence
   - The textual content of the retrieved knowledge subgraph
   - The structural patterns in the existing graph
   - The environmental report statistics
   This ensures semantic coherence with the label and existing nodes.

3. **Edge Connection Model**: When determining connections (neighbors) for new nodes, consider:
   - Semantic similarity between new node text and potential neighbor texts
   - Structural patterns in the existing graph
   - Balance between creating new connections and maintaining graph structure

IMPORTANT:
- You MUST strictly adhere to all of the above constraints.
- You MUST output the final enhanced graph data in **pure JSON format** with no additional explanations, annotations, or extra text.
- You MUST only answer the newly added nodes and should not repeat the graph data I gave you.
- When you are ready to answer, you MUST first output exactly the following line:  
  **"here are the generated datasets:"**  
  followed immediately by the JSON data, and nothing else.

Here is the graph data that needs augmentation:
{data_json_str}

You may think about the task step by step internally, but your final output MUST exactly follow the instructions above.
"""
        return prompt

    def _create_topological_enhancement_prompt(self, data_json_str, environment_state_str, target_label, label_target_size):
        """
        Creates a prompt for topological enhancement that models edge connections
        based on the paper's probability formula.
        """
        prompt = f"""# Graph Data Augmentation Task: Topological Enhancement

You are a Graph Data Enhancement Agent. We are focusing on label={target_label} because it is under-represented.
We would like to generate more nodes so that label={target_label} has more data.
The total target is about {label_target_size} or more, but do not exceed 5% new nodes overall.

## Topological enhancement focuses on:
- Improving the graph structure and connectivity 
- Balancing label distributions
- Enhancing community structures
- Creating meaningful connections between similar nodes

Please ensure that newly added nodes have label={target_label}, with mask="Train", 
and that any text modifications preserve semantic consistency with label={target_label}.
Please note that newly added nodes MUST have unique `new_node_id` values, starting from **"new_node 1"** and incrementing sequentially. For example, the first newly added nodes will be named "node_id": ""new_node 1"".

## Connection Probability Model:
When determining which existing nodes to connect to, implement this probability model:
P((v_s, v_i) ∈ E_c | K_t, R_t) = σ(θ₁·sim(x_s, x_i) + θ₂·|N(v_i) ∩ N_K(v_s)|/|N_K(v_s)| + θ₃·k_i/max_j k_j)

Where:
- sim(x_s, x_i) is the semantic similarity between node texts
- |N(v_i) ∩ N_K(v_s)|/|N_K(v_s)| represents neighborhood overlap
- k_i/max_j k_j is the normalized degree centrality
- σ is the sigmoid function

For topological enhancement, prioritize neighborhood overlap (θ₂) and degree centrality (θ₃) over semantic similarity (θ₁).

## Your augmentation tasks:
(1) Keep the original constraints: "text", "neighbors", "mask", etc.
(2) The newly added nodes must follow the format: {{new_node_id, label, text, neighbors, mask="Train"}}, 
    and cannot exceed 5% of the current node count.
(3) Focus on creating meaningful connections to existing nodes of the same label
(4) Output the final result in pure JSON.

Here is the environment state:
{environment_state_str}

The current partial data (for label={target_label}):
{data_json_str}

You MUST only answer the newly added nodes and should not repeat the graph data I gave you. When providing the final answer, you must first output the line:
"here are the generated datasets:"
and then the JSON data immediately. Nothing else.
"""
        return prompt

    def _apply_edge_probability_model(self, enhanced_data, original_data_str, mode):
        """
        Applies the edge probability model to the enhanced data according to the formula:
        P((v_s, v_i) ∈ E_c | K_t, R_t) = σ(θ₁·sim(x_s, x_i) + θ₂·|N(v_i) ∩ N_K(v_s)|/|N_K(v_s)| + θ₃·k_i/max_j k_j)
        
        Adjusts θ values based on the enhancement mode.
        """
        main_logger = logging.getLogger("main_logger")
        
        # If not a list, wrap in list
        if not isinstance(enhanced_data, list):
            if isinstance(enhanced_data, dict):
                enhanced_data = [enhanced_data]
            else:
                main_logger.error("[GraphEnhancementAgent] Enhanced data is not a valid list or dict")
                return enhanced_data
                
        # If empty, return as is
        if not enhanced_data:
            return enhanced_data
            
        try:
            # Parse original data to build the graph structure
            original_data = json.loads(original_data_str) if original_data_str else []
            if not isinstance(original_data, list):
                original_data = [original_data]
                
            # Create a graph from original data
            import networkx as nx
            G = nx.Graph()
            node_texts = {}  # Store node texts
            
            for node in original_data:
                if "node_id" not in node:
                    continue
                    
                node_id = node["node_id"]
                G.add_node(node_id)
                node_texts[node_id] = node.get("text", "")
                
                # Add edges
                neighbors = node.get("neighbors", [])
                for nbr in neighbors:
                    G.add_edge(node_id, nbr)
            
            # Get maximum degree for normalization
            max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
            
            # Set θ values based on mode
            if mode == "topological":
                theta_1 = 0.2  # Less weight on semantic similarity
                theta_2 = 0.5  # More weight on neighborhood overlap
                theta_3 = 0.3  # Moderate weight on degree centrality
            else:  # semantic mode
                theta_1 = 0.6  # More weight on semantic similarity
                theta_2 = 0.3  # Moderate weight on neighborhood overlap
                theta_3 = 0.1  # Less weight on degree centrality
            
            # Simplified semantic similarity function 
            def simple_sim(text1, text2):
                # Count overlapping words as a simple measure of similarity
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                return overlap / union if union > 0 else 0
            
            # Sigmoid function for probability
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            # Process each enhanced node
            for i, node in enumerate(enhanced_data):
                # Skip nodes that already have neighbors defined
                if "neighbors" in node and node["neighbors"]:
                    continue
                    
                node_text = node.get("text", "")
                if not node_text:
                    continue
                    
                new_neighbors = []
                
                # For each potential neighbor in the original graph
                for original_node_id in G.nodes():
                    # Calculate probability components
                    
                    # 1. Semantic similarity
                    similarity = simple_sim(node_text, node_texts.get(original_node_id, ""))
                    
                    # 2. Neighborhood overlap (simplified since new node has no neighbors yet)
                    # Use existing neighbors of enhanced nodes as a proxy
                    enhanced_neighbors = set()
                    for other_node in enhanced_data[:i]:  # Only consider previously processed nodes
                        if "neighbors" in other_node:
                            enhanced_neighbors.update(other_node["neighbors"])
                            
                    overlap_ratio = 0  # Default
                    if enhanced_neighbors:
                        original_neighbors = set(nx.neighbors(G, original_node_id))
                        overlap = len(original_neighbors.intersection(enhanced_neighbors))
                        overlap_ratio = overlap / len(enhanced_neighbors) if enhanced_neighbors else 0
                    
                    # 3. Degree centrality
                    degree_ratio = G.degree(original_node_id) / max_degree
                    
                    # Calculate overall probability
                    prob = sigmoid(theta_1 * similarity + theta_2 * overlap_ratio + theta_3 * degree_ratio)
                    
                    # Decide whether to add this neighbor based on probability
                    if random.random() < prob:
                        new_neighbors.append(original_node_id)
                
                # Ensure at least one neighbor (connect to most similar node if none selected)
                if not new_neighbors:
                    similarities = [(original_node_id, simple_sim(node_text, node_texts.get(original_node_id, "")))
                                   for original_node_id in G.nodes()]
                    most_similar = max(similarities, key=lambda x: x[1])[0] if similarities else None
                    if most_similar:
                        new_neighbors.append(most_similar)
                
                # Update neighbors in the enhanced node
                node["neighbors"] = new_neighbors
            
            return enhanced_data
            
        except Exception as e:
            main_logger.error(f"[GraphEnhancementAgent] Error applying edge probability model: {e}")
            # Return as is if error occurs
            return enhanced_data

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        Calls LLM for text generation using thread execution to avoid blocking.
        """
        result_dict = {}
        error_dict = {}

        def generate_output():
            try:
                output = self.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.90
                )
                result_dict["output"] = output[0]["generated_text"]
            except Exception as e:
                error_dict["error"] = str(e)

        main_logger = logging.getLogger("main_logger")
        gen_thread = threading.Thread(target=generate_output)
        gen_thread.start()

        while gen_thread.is_alive():
            time.sleep(0.5)
        gen_thread.join()

        if "error" in error_dict:
            main_logger.error(f"[GraphEnhancementAgent] Error during text generation: {error_dict['error']}")
            return "Error generating enhancement. Check logs for details."

        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extracts content after a specified flag in the text.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()