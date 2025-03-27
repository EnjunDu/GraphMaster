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
            prompt_for_enhance = f"""# Graph Data Augmentation Task: Topological Enhancement

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
        elif mode == "semantic":
            # 语义增强（原random模式）的 prompt
            prompt_for_enhance = f"""# Graph Data Augmentation Task: Semantic Enhancement

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

2. **Semantic enhancement overall task**: The mission of the enhancement task is to increase the diversity and comprehensiveness of the textual data while maintaining semantic coherence with existing nodes of the same label. You need to ensure that the text of the new node is comprehensive, diverse, and semantically rich. You need to understand the graph data I sent you as background knowledge, and think about how to generate the text and neighbors of the new node (the neighbors can consider the node IDs I sent you and their neighbors, and the text needs to be synthesized based on the background knowledge, but it cannot be completely consistent)

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