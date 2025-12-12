# enhancement_agent.py
import json
import time
import threading
import numpy as np
import random
import re
from transformers import TextGenerationPipeline
import logging


class GraphEnhancementAgent:
    """
    Enhancement Agent - implements the generation component of the RAG paradigm
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
        """
        main_logger = logging.getLogger("main_logger")
        
        # Create prompt based on mode
        if mode == "topological" and target_label:
            prompt_for_enhance = self._create_topological_enhancement_prompt(
                data_json_str, environment_state_str, target_label, label_target_size)
        else:  # mode == "semantic"
            prompt_for_enhance = self._create_semantic_enhancement_prompt(
                data_json_str, environment_state_str)

        main_logger.info(f"[GraphEnhancementAgent] LLM begin to enhance the given graph data with mode={mode}.")
        
        # **Generate enhanced data using LLM**
        try:
            # _call_generation already returns content without the prompt
            raw_output = self._call_generation(prompt_for_enhance, self.max_new_tokens)
            
            main_logger.info(f"[GraphEnhancementAgent] Generated output length: {len(raw_output)}")
            main_logger.info(f"[GraphEnhancementAgent] Generated output (first 1000 chars):\n{raw_output[:1000]}")
            
        except Exception as e:
            main_logger.error(f"[GraphEnhancementAgent] Error calling LLM generation: {e}")
            import traceback
            main_logger.error(f"[GraphEnhancementAgent] Traceback:\n{traceback.format_exc()}")
            return "[]"
        
        # Check if raw_output is valid
        if not raw_output or raw_output.strip() == "":
            main_logger.error(f"[GraphEnhancementAgent] LLM returned empty response")
            return "[]"
        
        # Extract JSON from LLM output
        splitted_flag = "here are the generated datasets:"
        enhanced_json_str = self._extract_after_flag(raw_output, splitted_flag)
        
        # 添加提取结果日志
        main_logger.info(f"[GraphEnhancementAgent] Extracted JSON length: {len(enhanced_json_str)}")
        if enhanced_json_str and enhanced_json_str != "[]":
            main_logger.info(f"[GraphEnhancementAgent] Extracted JSON (first 500 chars):\n{enhanced_json_str[:500]}")
        
        if not enhanced_json_str.strip() or enhanced_json_str == "[]":
            main_logger.warning("[GraphEnhancementAgent] Warning: No valid JSON data extracted from LLM output.")
            return "[]"
        
        # Postprocess the enhanced data
        try:
            # Parse the enhanced data
            enhanced_data = json.loads(enhanced_json_str)
            
            # Validate that we got actual data, not placeholder
            if isinstance(enhanced_data, list) and len(enhanced_data) > 0:
                # Check if first node has placeholder values
                first_node = enhanced_data[0]
                if first_node.get("text") == "...":
                    main_logger.warning("[GraphEnhancementAgent] LLM returned placeholder data, not actual content")
                    return "[]"
            
            # Apply mode-specific edge probability model
            enhanced_data = self._apply_edge_probability_model(enhanced_data, data_json_str, mode)
            
            # Convert back to string
            enhanced_json_str = json.dumps(enhanced_data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            main_logger.error(f"[GraphEnhancementAgent] Error parsing enhanced data: {e}")
            main_logger.error(f"[GraphEnhancementAgent] Problematic JSON: {enhanced_json_str[:500]}")
            return "[]"
        except Exception as e:
            main_logger.error(f"[GraphEnhancementAgent] Unexpected error in postprocessing: {e}")
            import traceback
            main_logger.error(f"[GraphEnhancementAgent] Traceback:\n{traceback.format_exc()}")
            return "[]"
        
        return enhanced_json_str

    def _create_semantic_enhancement_prompt(self, data_json_str, environment_state_str):
        """Creates a prompt for semantic enhancement"""
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

1. **New Node Addition**: You MUST add new nodes to enhance data diversity. Each new node MUST follow the same format as existing nodes: {{new_node_id, label, text, neighbors, mask}}. The mask for every new node MUST be **Train**. The total number of new nodes added MUST NOT exceed 10% of the total number of input nodes. Newly added nodes MUST have unique `new_node_id` values, starting from **"new_node_1"** and incrementing sequentially.

2. **Generate Real Content**: Each new node MUST have meaningful text content. DO NOT use placeholders like "...", "sample text", or similar. Generate actual, topic-relevant text.

3. **Edge Connections**: Assign appropriate neighbors to new nodes based on semantic similarity and structural patterns.

IMPORTANT OUTPUT FORMAT:
- You MUST output ONLY valid JSON - no explanations, no markdown code blocks
- Start directly with the flag "Here are the generated datasets:" followed by the JSON array
- Each node must have: node_id (string), label (integer), text (string), neighbors (array of strings), mask (string = "Train")

## Input Graph Data (JSON format):
{data_json_str}

Here are the generated datasets:"""
        return prompt

    def _create_topological_enhancement_prompt(self, data_json_str, environment_state_str, target_label, label_target_size):
        """Creates a prompt for topological enhancement"""
        prompt = f"""# Graph Data Augmentation Task: Topological Enhancement

You are a Graph Data Enhancement Agent. You are tasked with performing topological enhancement on graph data to address class imbalance. The graph nodes are described by the following structure: {{node_id, label, text, neighbors, mask}}.

## Topological enhancement focuses on:
- Balancing node distribution across different labels
- Improving graph structure and connectivity
- Creating strategic connections based on community structure
- Addressing class imbalance issues

## Current Environment State:
{environment_state_str}

## Target Enhancement:
- Target Label: {target_label}
- Current size for this label: {label_target_size}
- Goal: Generate new nodes with label={target_label} to balance the graph

## Your augmentation tasks are as follows:

1. **New Node Addition**: You MUST add new nodes with label={target_label}. Each new node MUST follow the format: {{new_node_id, label, text, neighbors, mask}}. The mask for every new node MUST be **Train**. The number of new nodes should help balance the class distribution. Newly added nodes MUST have unique `new_node_id` values, starting from **"new_node_1"** and incrementing sequentially.

2. **Generate Real Content**: Each new node MUST have meaningful text content relevant to label {target_label}. DO NOT use placeholders like "...", "sample text", or similar.

3. **Strategic Edge Connections**: Assign neighbors to new nodes based on:
   - Community structure patterns
   - Existing connectivity of nodes with label={target_label}
   - Modularity optimization principles

IMPORTANT OUTPUT FORMAT:
- You MUST output ONLY valid JSON - no explanations, no markdown code blocks
- Start directly with the flag "Here are the generated datasets:" followed by the JSON array
- Each node must have: node_id (string), label (integer = {target_label}), text (string), neighbors (array of strings), mask (string = "Train")

## Input Graph Data (JSON format - sample nodes with label={target_label}):
{data_json_str}

Here are the generated datasets:"""
        return prompt

    def _call_generation(self, prompt, max_new_tokens):
        """
        Call the text generation pipeline with timeout protection
        """
        main_logger = logging.getLogger("main_logger")
        result = [None]
        exception = [None]
        
        def target():
            try:
                # **不使用 return_full_text 参数**
                outputs = self.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.text_generation.tokenizer.eos_token_id
                )
                result[0] = outputs
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=300)  # 5 minutes timeout
        
        if thread.is_alive():
            main_logger.error("[GraphEnhancementAgent] LLM generation timed out after 300 seconds")
            return ""
        
        if exception[0]:
            main_logger.error(f"[GraphEnhancementAgent] LLM generation failed: {exception[0]}")
            return ""
        
        if result[0] is None or len(result[0]) == 0:
            main_logger.error("[GraphEnhancementAgent] LLM returned no output")
            return ""
        
        # **手动提取生成的文本**
        full_output = result[0][0]['generated_text']
        
        # **关键：去除提示词部分**
        # 如果输出包含完整提示词，则去除
        if full_output.startswith(prompt):
            generated_only = full_output[len(prompt):].strip()
            main_logger.info(f"[GraphEnhancementAgent] Removed prompt prefix, generated text length: {len(generated_only)}")
            return generated_only
        else:
            # 如果不是以提示词开头，返回完整输出（可能 pipeline 已经去除了）
            main_logger.info(f"[GraphEnhancementAgent] Full output length: {len(full_output)}")
            return full_output

    def _extract_after_flag(self, text, flag):
        """
        Extracts JSON content after a specific flag in the text.
        """
        main_logger = logging.getLogger("main_logger")
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        flag_lower = flag.lower()
        
        if flag_lower in text_lower:
            idx = text_lower.find(flag_lower)
            after_flag = text[idx + len(flag):].strip()
            
            # Try to find the start of JSON array
            json_start = after_flag.find('[')
            if json_start != -1:
                after_flag = after_flag[json_start:]
                
                # Try to find the matching closing bracket
                try:
                    # Use a simple bracket counting method
                    bracket_count = 0
                    for i, char in enumerate(after_flag):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                return after_flag[:i+1]
                    
                    # If we didn't find a matching bracket, return everything
                    return after_flag
                except Exception as e:
                    main_logger.warning(f"[GraphEnhancementAgent] Error finding JSON end: {e}")
                    return after_flag
            else:
                return after_flag
        else:
            # If flag not found, try to extract JSON directly
            json_start = text.find('[')
            if json_start != -1:
                return text[json_start:]
            return text

    def _apply_edge_probability_model(self, enhanced_data, original_data_str, mode):
        """
        Apply edge probability model to adjust neighbor connections.
        """
        main_logger = logging.getLogger("main_logger")
        
        try:
            # Parse original data to get existing nodes
            original_data = json.loads(original_data_str)
            existing_node_ids = {node['node_id'] for node in original_data}
            
            # Calculate edge statistics from original data
            total_edges = sum(len(node.get('neighbors', [])) for node in original_data)
            avg_degree = total_edges / len(original_data) if original_data else 0
            
            main_logger.info(f"[GraphEnhancementAgent] Original graph: {len(original_data)} nodes, avg degree: {avg_degree:.2f}")
            
            # Process each enhanced node
            for node in enhanced_data:
                current_neighbors = node.get('neighbors', [])
                
                if mode == "semantic":
                    # For semantic mode: moderate edge probability, focus on semantic similarity
                    edge_prob = 0.3
                    target_degree = max(2, int(avg_degree * 0.8))
                else:  # topological mode
                    # For topological mode: higher edge probability, focus on structure
                    edge_prob = 0.5
                    target_degree = max(3, int(avg_degree * 1.2))
                
                # Filter neighbors to only include existing nodes
                valid_neighbors = [n for n in current_neighbors if n in existing_node_ids]
                
                # If we have too few neighbors, try to add more from original data
                if len(valid_neighbors) < target_degree:
                    # Sample additional neighbors from original data
                    potential_neighbors = list(existing_node_ids - set(valid_neighbors))
                    
                    if potential_neighbors:
                        # For topological mode, prefer nodes with same label
                        if mode == "topological" and 'label' in node:
                            same_label_nodes = [
                                orig_node['node_id'] 
                                for orig_node in original_data 
                                if orig_node.get('label') == node['label'] 
                                and orig_node['node_id'] in potential_neighbors
                            ]
                            if same_label_nodes:
                                potential_neighbors = same_label_nodes + [
                                    n for n in potential_neighbors if n not in same_label_nodes
                                ]
                        
                        # Add neighbors based on probability
                        num_to_add = min(
                            target_degree - len(valid_neighbors),
                            len(potential_neighbors)
                        )
                        
                        for neighbor in potential_neighbors[:num_to_add * 2]:  # Consider 2x candidates
                            if len(valid_neighbors) >= target_degree:
                                break
                            if random.random() < edge_prob:
                                valid_neighbors.append(neighbor)
                
                # Update neighbors
                node['neighbors'] = valid_neighbors[:target_degree * 2]  # Cap at 2x target
                
            main_logger.info(f"[GraphEnhancementAgent] Applied {mode} edge probability model to {len(enhanced_data)} nodes")
            
        except Exception as e:
            main_logger.error(f"[GraphEnhancementAgent] Error in edge probability model: {e}")
            import traceback
            main_logger.error(f"[GraphEnhancementAgent] Traceback:\n{traceback.format_exc()}")
        
        return enhanced_data