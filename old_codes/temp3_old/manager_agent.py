import json
import os
import time
import threading
from transformers import TextGenerationPipeline
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent
import logging


class ManagerAgent:
    """
    管理者Agent，用于：
      1) 调度 PerceptionAgent 进行图感知（环境报告）与采样（由LLM决策）
      2) 加载并筛选原始数据（初始输入）
      3) 自动决定本轮应使用的增强模式（semantic或topological）
      4) 执行单次增强-评估迭代，并返回两个输出：
         - evaluated_data_str：评估后的图数据（JSON字符串）
         - continue_flag：布尔值，True 表示需要继续增强，False 表示停止
    """

    def __init__(self,
                 text_generation_pipeline: TextGenerationPipeline,
                 perception_agent: GraphPerceptionAgent,
                 enhancement_agent: GraphEnhancementAgent,
                 evaluation_agent: GraphEvaluationAgent,
                 data_file: str,
                 visualize_sampling: bool = False,
                 enhancement_mode: str = None,  # 可以为None，表示自动决定
                 target_label: int = None,
                 label_target_size: int = 0
                 ):
        """
        :param text_generation_pipeline: 大语言模型的文本生成 pipeline
        :param perception_agent: 感知Agent
        :param enhancement_agent: 增强Agent
        :param evaluation_agent: 评估Agent
        :param data_file: 包含节点信息的 JSON 文件路径
        :param visualize_sampling: 是否在感知阶段可视化采样结果
        :param enhancement_mode: 'semantic' or 'topological' or None (自动决定)
        :param target_label: 当 enhancement_mode='topological' 时，需要增强的标签名
        :param label_target_size: 希望该标签达到的总数据量
        """
        self.text_generation = text_generation_pipeline

        self.perception_agent = perception_agent
        self.enhancement_agent = enhancement_agent
        self.evaluation_agent = evaluation_agent
        self.data_file = data_file
        self.visualize_sampling = visualize_sampling
        self.enhancement_mode = enhancement_mode  # 可能为None
        self.target_label = target_label
        self.label_target_size = label_target_size

        # 保存初始环境报告（增强前的基准状态）
        main_logger = logging.getLogger("main_logger")
        try:
            self.initial_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True  # 总是获取标签分布，用于自动决定增强模式
            )
            main_logger.info("[ManagerAgent] Initial environment report generated and stored.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating initial environment report: {e}")
            self.initial_environment_report = json.dumps({"error": "Failed to generate initial report"})

        # 如果enhancement_mode为None，则自动决定初始的增强模式
        if self.enhancement_mode is None:
            try:
                self.enhancement_mode = self.decide_enhancement_mode(self.initial_environment_report)
                main_logger.info(f"[ManagerAgent] Auto-decided initial enhancement mode: {self.enhancement_mode}")
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error deciding enhancement mode: {e}")
                self.enhancement_mode = "semantic"  # 默认使用semantic模式
                main_logger.info(f"[ManagerAgent] Falling back to default enhancement mode: {self.enhancement_mode}")

        # 在初始化阶段，不做采样
        self.sampled_node_ids = []

    def load_initial_data(self) -> str:
        """
        从 data_file 中加载数据，并根据 sampled_node_ids 筛选出需要处理的节点，
        返回 JSON 字符串。
        """
        main_logger = logging.getLogger("main_logger")  # 获取 main_logger
        if not os.path.exists(self.data_file):
            main_logger.warning(f"[ManagerAgent] data_file={self.data_file} not found.")
            return ""
        with open(self.data_file, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        main_logger.info(f"[ManagerAgent] Loaded dataset from {self.data_file}, total nodes: {len(full_dataset)}")

        if not self.sampled_node_ids:
            main_logger.warning("[ManagerAgent] No sampled_node_ids found. Return empty string.")
            return ""

        selected_data = [node for node in full_dataset if node.get("node_id") in self.sampled_node_ids]
        if not selected_data:
            main_logger.error("[ManagerAgent] No matching nodes in dataset.")
            return ""
        main_logger.info(f"[ManagerAgent] Found {len(selected_data)} nodes in dataset for enhancement-evaluation.")
        return json.dumps(selected_data, ensure_ascii=False, indent=2)

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        调用LLM生成文本，使用线程执行以避免阻塞，增加错误处理和使用greedy decoding
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
            main_logger.error(f"[ManagerAgent] Error during text generation: {error_dict['error']}")
            return "Error in manager decision. Check logs for details."

        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        从 text 提取 "here are the generated datasets:" 之后的内容，并进行 JSON 修正
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""

        extracted_json = text[idx + len(flag):].strip()

        # 修正 JSON（去除非法控制字符）
        try:
            extracted_json = extracted_json.replace("\\n", "").replace("\\t", "").replace('\\"', '"')
            json.loads(extracted_json)  # 确保 JSON 解析不会失败
            return extracted_json
        except json.JSONDecodeError:
            main_logger = logging.getLogger("main_logger")  # 获取 main_logger
            main_logger.error("[EvaluationAgent] Warning: JSON decoding failed, returning empty string.")
            return ""

    def decide_enhancement_mode(self, environment_report_str) -> str:
        """
        根据环境报告自动决定应该使用哪种增强模式：semantic或topological

        决策依据：
        1. 分析社区聚集程度 - 如果社区结构不清晰，倾向于topological增强
        2. 分析标签分布 - 如果标签分布不平衡，倾向于topological增强
        3. 综合考虑图的整体特性

        返回 'semantic' 或 'topological'
        """
        main_logger = logging.getLogger("main_logger")
        main_logger.info("[ManagerAgent] Deciding enhancement mode based on environment report...")

        # 解析环境报告
        try:
            env_report = json.loads(environment_report_str)
        except json.JSONDecodeError:
            main_logger.error(
                "[ManagerAgent] Failed to parse environment report JSON. Defaulting to semantic enhancement.")
            return "semantic"

        # 构建决策提示
        prompt = f"""You are a Graph Analysis Expert. 

Based on the following environment report of a graph:
{environment_report_str}

You need to decide which enhancement mode would be more beneficial for this graph:
1. "semantic" enhancement - focuses on improving node attributes and content quality
2. "topological" enhancement - focuses on graph structure and label distribution

Factors to consider:
- Community structure clarity and distribution
- Label distribution balance/imbalance
- Node connectivity patterns
- Current graph state health

Based solely on the environment report, which enhancement mode would be more beneficial right now?
Please output ONLY the word "semantic" OR "topological" without any explanation or additional text.
"""

        # 调用大语言模型获取决策
        decision = self._call_generation(prompt, max_tokens=100).strip().lower()

        # 解析结果
        if "topological" in decision:
            mode = "topological"
        else:
            # 默认为语义增强
            mode = "semantic"

        main_logger.info(f"[ManagerAgent] Enhancement mode decision: {mode}")

        # 如果选择了拓扑增强但没有指定目标标签，需要自动决定
        if mode == "topological" and self.target_label is None:
            self.target_label = self.decide_target_label(environment_report_str)
            main_logger.info(
                f"[ManagerAgent] Auto-selected target label for topological enhancement: {self.target_label}")

        return mode

    def decide_target_label(self, environment_report_str) -> int:
        """
        根据环境报告自动决定用于拓扑增强的目标标签
        通常选择样本量最少或分布不均衡的标签
        """
        main_logger = logging.getLogger("main_logger")

        try:
            env_report = json.loads(environment_report_str)
            if "LabelDistribution" not in env_report:
                main_logger.warning(
                    "[ManagerAgent] No label distribution in environment report. Defaulting to label 0.")
                return 0

            label_dist = env_report["LabelDistribution"]
            # 找出样本量最少的标签
            min_label = min(label_dist.items(), key=lambda x: int(x[1]))
            target = int(min_label[0]) if min_label[0].isdigit() else 0

            # 如果没有设置目标大小，设置为平均值的80%
            if self.label_target_size == 0:
                avg_size = sum(int(count) for count in label_dist.values()) / len(label_dist)
                self.label_target_size = int(avg_size * 0.8)

            return target

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            main_logger.error(f"[ManagerAgent] Error deciding target label: {e}. Defaulting to label 0.")
            return 0

    def run_manager_pipeline(self, early_stopping, current_iteration, current_data_str: str = None) -> (str, bool):
        """
        执行单次增强-评估迭代：
          1) 先获取最新的环境报告
          2) 自动决定本轮使用的增强模式（semantic或topological）
          3) 决定采样 (sampled_node_ids)
          4) 如果未提供 current_data_str，则加载初始数据
          5) 调用增强Agent 与 评估Agent
          6) 让LLM判断是否继续迭代
        """
        main_logger = logging.getLogger("main_logger")  # 获取 main_logger

        # 1) 获取当前环境报告（本轮增强前的状态）
        try:
            current_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True  # 总是获取标签分布，用于自动决定增强模式
            )
            main_logger.info("[ManagerAgent] Generated current environment report for iteration.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating current environment report: {e}")
            return "", False

        # 2) 在每轮自动决定增强模式，除非命令行已指定固定模式
        if self.enhancement_mode is None or current_iteration % 3 == 0:  # 每3轮或初始时重新决定
            try:
                prev_mode = self.enhancement_mode
                self.enhancement_mode = self.decide_enhancement_mode(current_environment_report)
                if prev_mode != self.enhancement_mode:
                    main_logger.info(
                        f"[ManagerAgent] Enhancement mode changed from {prev_mode} to {self.enhancement_mode}")
                else:
                    main_logger.info(f"[ManagerAgent] Enhancement mode remains {self.enhancement_mode}")
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error deciding enhancement mode: {e}")
                if self.enhancement_mode is None:
                    self.enhancement_mode = "semantic"  # 默认使用semantic模式
                    main_logger.info(f"[ManagerAgent] Setting default enhancement mode: {self.enhancement_mode}")

        # 3) 根据当前模式决定如何采样:
        try:
            if self.enhancement_mode == "semantic":
                self.sampled_node_ids = self.perception_agent.decide_sampling(visualize=self.visualize_sampling)
            else:  # topological mode
                self.sampled_node_ids = self.perception_agent.decide_label_sampling(
                    target_label=self.target_label,
                    label_target_size=self.label_target_size,
                    visualize=self.visualize_sampling
                )
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during sampling: {e}")
            self.sampled_node_ids = []

        if not self.sampled_node_ids:
            main_logger.warning("[ManagerAgent] No nodes sampled. Stopping iteration.")
            return "", False

        # 4) 如果没有提供 current_data_str，就从文件里过滤出需要的节点
        if current_data_str is None:
            current_data_str = self.load_initial_data()
            if not current_data_str:
                main_logger.warning("[ManagerAgent] current_data_str is empty. Stopping iteration.")
                return "", False

        # 5) 增强阶段
        try:
            enhanced_data_str = self.enhancement_agent.enhance_graph(
                data_json_str=current_data_str,
                environment_state_str=current_environment_report,  # 使用当前环境报告
                mode=self.enhancement_mode,
                target_label=self.target_label,
                label_target_size=self.label_target_size
            )

            if not enhanced_data_str.strip():
                main_logger.error("[ManagerAgent] Enhancement result is empty, stopping iteration.")
                return "", False

            main_logger.info(f'[ManagerAgent] Enhanced data generated successfully.')
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during enhancement: {e}")
            return "", False

        # 6) 评估阶段 - 传入初始环境报告、当前环境报告和原始数据
        try:
            evaluated_data_str = self.evaluation_agent.evaluate_graph(
                original_data_str=current_data_str,
                generated_data_str=enhanced_data_str,
                initial_environment_report=self.initial_environment_report,
                current_environment_report=current_environment_report,
                mode=self.enhancement_mode,
                target_label=self.target_label,
                perception_agent=self.perception_agent  # 传递perception_agent以便重新生成报告
            )

            if not evaluated_data_str.strip():
                main_logger.error("[ManagerAgent] Evaluation result is empty, stopping iteration.")
                return "", False
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during evaluation: {e}")
            return "", False

        # 7) 通过 LLM 判断是否继续增强 (若已到 early_stopping 轮，则启用 LLM 决策)
        if current_iteration < early_stopping:
            # 强制继续
            continue_flag = True
        else:
            try:
                # 生成一个新的 prompt，让LLM比较初始和当前环境状态
                if self.enhancement_mode == "topological":
                    manager_prompt_mode_part = f"We are specifically performing few-shot topological augmentation for label={self.target_label}."
                else:
                    manager_prompt_mode_part = f"We are performing semantic augmentation."

                prompt_manager = f"""You are a Manager Agent. 
{manager_prompt_mode_part}

## Initial environment report (before enhancement):
{self.initial_environment_report}

## Current environment report (after this round):
{current_environment_report}

## Generated data:
{enhanced_data_str}

## Evaluation result:
{evaluated_data_str}

Please carefully analyze the differences between the initial and current environment reports.
Based on these changes and the evaluation results, determine if we should continue the enhancement process.

Output True if you see significant potential for further enhancement.
Output False if you detect convergence or minimal improvement.

When providing the final answer, you must say "here is the Final decision:" and then output True or False immediately without extra text.
"""

                decision_output = self._call_generation(prompt_manager, max_tokens=4196)
                decision_str = self._extract_after_flag(decision_output, "here is the Final decision:")
                if decision_str.lower() == "true":
                    continue_flag = True
                else:
                    continue_flag = False
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error during continuation decision: {e}")
                # 默认停止迭代
                continue_flag = False

        return evaluated_data_str, continue_flag