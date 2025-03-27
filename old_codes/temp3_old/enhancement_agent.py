# enhancement_agent.py
import json
import time
import threading
from transformers import TextGenerationPipeline
import logging


class GraphEnhancementAgent:
    """
    用于执行图增强任务的 Agent。
    利用传入的 TextGenerationPipeline，根据 Prompt 生成增强后的图数据（JSON格式）。
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: 文本生成 pipeline（已加载好的 LLM）
        :param max_new_tokens: 每次生成的最大 token 数
        """
        self.text_generation = text_generation_pipeline
        self.max_new_tokens = max_new_tokens

    def enhance_graph(self, data_json_str: str, environment_state_str: str = "",
                      mode: str = "semantic",
                      target_label: str = None,
                      label_target_size: int = 0
                      ) -> str:
        """
        执行图增强任务，返回增强后的图数据（JSON字符串）。
        如果 mode='topological'，则在 prompt 中说明正在进行小样本标签增强。

        :param data_json_str: 输入的图数据（JSON字符串）
        :param environment_state_str: 环境状态报告
        :param mode: 增强模式，'semantic'或'topological'
        :param target_label: topological模式下的目标标签
        :param label_target_size: topological模式下的目标数量
        :return: 增强后的图数据（JSON字符串）
        """
        if mode == "topological" and target_label:
            # 拓扑增强（原label模式）的提示
            prompt_for_enhance = f"""# Text Attribute Graph Data Augmentation Task: Background Knowledge Enhancement

    You are a TAG Data Synthesis Agent. We focus on label={target_label} because it is under-represented. We would like to generate more nodes to enrich the data for label={target_label}. The total target is about {label_target_size} or more, but do not exceed 5% new nodes overall.

    ## Background Knowledge Enhancement Focus:
    - Enhance the textual content by incorporating relevant background knowledge to increase its informational value.
    - Augment the text with domain-specific descriptive knowledge while maintaining semantic consistency with label={target_label}.
    - Leverage background knowledge to create meaningful connections with existing nodes of the same label, thus balancing the label distribution.
    - Establish connections based on content relevance rather than solely on topological improvements.

    Please ensure that newly added nodes have label={target_label} and that the mask is set to "Train". Any text modifications must preserve semantic consistency with label={target_label}. Newly added nodes must have unique `new_node_id` values, starting from **"new_node 1"** and incrementing sequentially. For example, the first new node should be named "node_id": "new_node 1".

    ## Your Augmentation Tasks:
    1. Maintain the original constraints: including properties like "text", "neighbors", "mask", etc.
    2. The newly added nodes must follow the format:  
    ```json
    [
        {"node_id": "new_node_1", "label": 5, "text": "the newly generated text with background knowledge", "neighbors": [123, 456], "mask": "Train"},
        {"node_id": "new_node_2", "label": 5, "text": "the newly generated text with background knowledge", "neighbors": [234, 567], "mask": "Train"}
    ]
    The number of newly added nodes must not exceed 5% of the current node count. 
    3. Focus on leveraging background knowledge to create meaningful connections to existing nodes with the same label. 
    4. The final result must be output in pure JSON format.

    Environment state: {environment_state_str}

    Current partial data (for label={target_label}): {data_json_str}

    You MUST only output the newly added node data and should not repeat the graph data I provided. When providing the final answer, you must first output the line: "here are the generated datasets:"
    and then immediately output the JSON data, and nothing else.
        """
        elif mode == "semantic":
            # 语义增强（原random模式）的 prompt
            prompt_for_enhance = f"""# Graph Data Augmentation Task: Semantic Enhancement

You are a Graph Data Enhancement Agent. Your task is to perform semantic enhancement on graph data. The graph nodes are described by the following structure: {node_id, label, text, neighbors, mask}. The `mask` field can have one of the following values: **Train**, **Test**, or **Validation**.

## Semantic Enhancement Focus:
- Improve data quality and diversity.
- Enhance the meaning and content of nodes.
- Add variations to existing semantic patterns.
- Enrich the textual information while ensuring label consistency.

## Relevant Environment State:
{environment_state_str}

## Your Augmentation Tasks:

1. **New Node Addition**:
   - You MUST add new nodes to enhance data diversity.
   - Each new node MUST follow the same format as existing nodes: {new_node_id, label, text, neighbors, mask}.
   - The mask for every new node MUST be **Train**.
   - The total number of new nodes added MUST NOT exceed 10% of the total number of input nodes (if 10% is fractional, round down to the nearest integer).
   - Newly added nodes MUST have unique `new_node_id` values, starting from **"new_node 1"** and incrementing sequentially.
   
   Example:
   ```json
   [
     {"node_id": "new_node_1", "label": 5, "text": "the newly generated text", "neighbors": [123, 456], "mask": "Train"},
     {"node_id": "new_node_2", "label": 5, "text": "the newly generated text", "neighbors": [234, 567], "mask": "Train"}
   ]
Semantic Enhancement Overall Task:

Increase the diversity and comprehensiveness of the textual data while maintaining semantic coherence with existing nodes of the same label.

Ensure that the text for each new node is comprehensive, diverse, and semantically rich.

Leverage the background information provided by the graph data to synthesize new text and determine meaningful neighbors. While the new text should be based on the background knowledge, it must not be entirely consistent with the existing node texts.

IMPORTANT:

- You MUST strictly adhere to all of the above constraints.

- You MUST output the final enhanced graph data in pure JSON format with no additional explanations, annotations, or extra text.

- You MUST only output the newly added nodes and should not repeat the graph data I provided.

- When you are ready to answer, you MUST first output exactly the following line: "here are the generated datasets:" followed immediately by the JSON data, and nothing else.

Here is the graph data that needs augmentation: {data_json_str}

You may think about the task step by step internally, but your final output MUST exactly follow the instructions above.
"""

        main_logger = logging.getLogger("main_logger")
        main_logger.info(f"[GraphEnhancementAgent] LLM begin to enhance the given graph data with mode={mode}.")

        raw_output = self._call_generation(prompt_for_enhance, self.max_new_tokens)
        splitted_flag = "here are the generated datasets:"
        enhanced_json_str = self._extract_after_flag(raw_output, splitted_flag)
        if not enhanced_json_str.strip():
            main_logger.warning("[GraphEnhancementAgent] Warning: LLM returned empty enhancement result.")
        return enhanced_json_str

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        使用LLM生成文本，使用线程执行以避免阻塞，增加错误处理和使用greedy decoding
        """
        result_dict = {}
        error_dict = {}

        def generate_output():
            try:
                output = self.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=True,  # 改为False，使用greedy decoding避免采样问题
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
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()