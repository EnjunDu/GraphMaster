import json
import time
import threading
from transformers import TextGenerationPipeline
import logging

class GraphEnhancementAgent:
    """
    Agent responsible for performing graph data enhancement.
    Given the current graph data and environment state, it uses the LLM (via prompt) to generate new nodes (in JSON format).
    Enhancement modes:
      - Semantic: Focus on enriching node content and adding diverse nodes.
      - Topological: Focus on adding nodes to improve graph structure and balance label distribution.
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: LLM text generation pipeline (pre-loaded model).
        :param max_new_tokens: Maximum tokens to generate for augmentation.
        """
        self.text_generation = text_generation_pipeline
        self.max_new_tokens = max_new_tokens

    def enhance_graph(self, data_json_str: str, environment_state_str: str = "",
                      mode: str = "semantic", target_label: str = None, label_target_size: int = 0) -> str:
        """
        Perform graph enhancement and return the augmented graph data as a JSON string.
        If mode='topological', prompts the LLM to focus on augmenting the under-represented label.
        In both modes, the agent will generate new nodes with appropriate attributes and edges.
        :param data_json_str: JSON string of the current graph data to augment (usually the sampled subgraph or relevant subset).
        :param environment_state_str: The environment report (graph state summary) to guide augmentation.
        :param mode: 'semantic' or 'topological' enhancement mode.
        :param target_label: If topological mode, the label that needs augmentation (more nodes).
        :param label_target_size: If topological mode, the desired total count for target_label (used as guidance, not strict).
        :return: JSON string of only the new nodes generated (to be merged with the original data elsewhere).
        """
        # Build the prompt based on the mode
        if mode == "topological" and target_label is not None:
            # Prompt for topological augmentation focusing on a specific label
            prompt_for_enhance = f"""# Graph Data Augmentation Task: Topological Enhancement

You are a Graph Data Enhancement Agent. We need to augment the graph's topology, focusing on label={target_label} (currently under-represented).
We aim to add nodes such that label={target_label} is better represented, targeting roughly {label_target_size} total nodes for this label (but do not exceed ~5% new nodes overall).

## Topological enhancement focuses on:
- Improving graph structure and connectivity.
- Balancing label distributions.
- Strengthening community structure by adding nodes that connect to existing nodes of label={target_label}.

Guidelines for new nodes:
- Each new node must have label={target_label}, mask="Train".
- Newly added nodes should connect meaningfully to existing nodes of the same label (to integrate into the graph).
- Preserve semantic consistency: new nodes' "text" should be plausible for label={target_label}.
- Use `new_node_id` for each new node (unique, starting from "new_node 1", "new_node 2", ...).

## Output Format Requirements:
- Only output the new nodes as a JSON list (do NOT include the original nodes).
- Each new node entry: {new_node_id, label, text, neighbors, mask="Train"}.
- Limit the number of new nodes to at most 5% of current node count (for safety).

## Current Environment State:
{environment_state_str}

## Current Partial Graph Data (label={target_label} subset):
{data_json_str}

When you provide the final answer, start with:
"here are the generated datasets:"
Followed immediately by the JSON array of the new nodes.
"""
        else:  # semantic mode
            # Prompt for semantic augmentation focusing on content diversity
            prompt_for_enhance = f"""# Graph Data Augmentation Task: Semantic Enhancement

You are a Graph Data Enhancement Agent. Your task is to enrich the graph data semantically.
Each graph node has the format: {{node_id, label, text, neighbors, mask}}, where mask ∈ {{Train, Test, Validation}}.

## Semantic enhancement focuses on:
- Improving data quality and diversity.
- Enriching node text content while maintaining label consistency.
- Adding variations to semantic content without altering label semantics.

Guidelines for new nodes:
1. **New Node Addition**: Add new nodes to increase data diversity. Each new node must follow the same format: {{new_node_id, label, text, neighbors, mask}}. The mask for all new nodes is "Train".
   - Limit new nodes to ≤ 10% of input node count (round down if fractional).
   - Use unique `new_node_id` starting from "new_node 1", "new_node 2", ... sequentially.
2. **Semantic Quality**: The text of each new node should be rich, diverse, and coherent with its label's context (not duplicate existing text).
   - Use the existing graph data as background knowledge for content creation.
   - Ensure neighbors of a new node connect logically to existing nodes (by ID) that share context or label.
3. **Consistency**: Maintain semantic coherence with the label. The new node's text should clearly relate to the label's theme but introduce new content or perspective.

## Relevant Environment State:
{environment_state_str}

## Graph Data to Augment (context):
{data_json_str}

IMPORTANT:
- Adhere strictly to format and constraints.
- Output only the newly added nodes in pure JSON format, no extra commentary.
- Begin your output with: "here are the generated datasets:" and then the JSON array of new nodes.
"""
        main_logger = logging.getLogger("main_logger")
        main_logger.info(f"[GraphEnhancementAgent] Generating enhancement in mode={mode}, target_label={target_label}.")

        # Use the LLM to generate the augmented data
        raw_output = self._call_generation(prompt_for_enhance, self.max_new_tokens)
        splitted_flag = "here are the generated datasets:"
        new_nodes_json_str = self._extract_after_flag(raw_output, splitted_flag)
        if not new_nodes_json_str.strip():
            main_logger.warning("[GraphEnhancementAgent] Warning: LLM returned empty enhancement result.")
        return new_nodes_json_str

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        Internal helper to invoke LLM generation using a separate thread to avoid blocking.
        Uses greedy decoding to reduce randomness.
        """
        result_dict = {}
        error_dict = {}

        def generate_output():
            try:
                output = self.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.7,
                    top_p=1
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
        Extract everything that comes after a certain flag string in 'text', typically to isolate JSON content.
        If the flag is not found, returns an empty string.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()
