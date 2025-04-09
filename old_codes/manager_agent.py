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
    Manager Agent, used to:
    1) Schedule PerceptionAgent for graph perception (environmental reporting) and sampling (decided by LLM)
    2) Load and filter raw data (initial input)
    3) Automatically determine the enhancement mode to be used in this round (semantic or topological)
    4) Perform a single enhancement-evaluation iteration and return two outputs:
        - evaluated_data_str: evaluated graph data (JSON string)
        - continue_flag: Boolean value, True means that enhancement needs to continue, False means stop
    """

    def __init__(self,
                 text_generation_pipeline: TextGenerationPipeline,
                 perception_agent: GraphPerceptionAgent,
                 enhancement_agent: GraphEnhancementAgent,
                 evaluation_agent: GraphEvaluationAgent,
                 data_file: str,
                 visualize_sampling: bool = False,
                 enhancement_mode: str = None,  # Can be None, indicating automatic decision
                 target_label: int = None,
                 label_target_size: int = 0
                 ):
        """
        :param text_generation_pipeline: Text generation pipeline for large language models
        :param perception_agent: Perception Agent
        :param enhancement_agent: Enhancement Agent
        :param evaluation_agent: Evaluation Agent
        :param data_file: JSON file path containing node information
        :param visualize_sampling: Whether to visualize sampling results in the perception phase
        :param enhancement_mode: 'semantic' or 'topological' or None (automatically determined)
        :param target_label: When enhancement_mode='topological', the label name to be enhanced
        :param label_target_size: The total amount of data expected for this label
        """
        self.text_generation = text_generation_pipeline

        self.perception_agent = perception_agent
        self.enhancement_agent = enhancement_agent
        self.evaluation_agent = evaluation_agent
        self.data_file = data_file
        self.visualize_sampling = visualize_sampling
        self.enhancement_mode = enhancement_mode  # May be None
        self.target_label = target_label
        self.label_target_size = label_target_size

        # Save the initial environment report (baseline state before enhancement)
        self.initial_environment_report = self.perception_agent.generate_environment_report(
            require_label_distribution=True  # Always obtain label distribution for automatic decision of enhancement mode
        )
        main_logger = logging.getLogger("main_logger")
        main_logger.info("[ManagerAgent] Initial environment report generated and stored.")

        # If enhancement_mode is None, the initial enhancement mode is automatically determined
        if self.enhancement_mode is None:
            self.enhancement_mode = self.decide_enhancement_mode(self.initial_environment_report)
            main_logger.info(f"[ManagerAgent] Auto-decided initial enhancement mode: {self.enhancement_mode}")

        # During the initialization phase, no sampling is performed
        self.sampled_node_ids = []

    def load_initial_data(self) -> str:
        """
        Load data from data_file and filter out nodes to be processed based on sampled_node_ids,
        and return a JSON string.
        """
        main_logger = logging.getLogger("main_logger")  # Get main_logger
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
        result_dict = {}

        def generate_output():
            output = self.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=1.0
            )
            result_dict["output"] = output[0]["generated_text"]

        gen_thread = threading.Thread(target=generate_output)
        gen_thread.start()
        while gen_thread.is_alive():
            time.sleep(0.5)
        gen_thread.join()
        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extract the content after "here are the generated datasets:" from text and make JSON corrections
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""

        extracted_json = text[idx + len(flag):].strip()

        # Fix JSON (remove illegal control characters)
        try:
            extracted_json = extracted_json.replace("\\n", "").replace("\\t", "").replace('\\"', '"')
            json.loads(extracted_json)  # Make sure JSON parsing doesn't fail
            return extracted_json
        except json.JSONDecodeError:
            main_logger = logging.getLogger("main_logger")  # Get main_logger
            main_logger.error("[EvaluationAgent] Warning: JSON decoding failed, returning empty string.")
            return ""

    def decide_enhancement_mode(self, environment_report_str) -> str:
        """
        Automatically decide which enhancement mode should be used based on the environment report: semantic or topological

        Decision basis:
        1. Analyze the degree of community aggregation - if the community structure is not clear, tend to use topological enhancement
        2. Analyze the label distribution - if the label distribution is unbalanced, tend to use topological enhancement
        3. Comprehensively consider the overall characteristics of the graph

        Return 'semantic' or 'topological'
        """
        main_logger = logging.getLogger("main_logger")
        main_logger.info("[ManagerAgent] Deciding enhancement mode based on environment report...")

        # Analyze environmental reports
        try:
            env_report = json.loads(environment_report_str)
        except json.JSONDecodeError:
            main_logger.error("[ManagerAgent] Failed to parse environment report JSON. Defaulting to semantic enhancement.")
            return "semantic"

        # Build decision prompts
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

        # Calling the large language model to obtain the decision
        decision = self._call_generation(prompt, max_tokens=100).strip().lower()

        # Parse results
        if "topological" in decision:
            mode = "topological"
        else:
            # Default is semantic enhancement
            mode = "semantic"

        main_logger.info(f"[ManagerAgent] Enhancement mode decision: {mode}")

        # If topology enhancement is selected but no target label is specified, it needs to be automatically determined
        if mode == "topological" and self.target_label is None:
            self.target_label = self.decide_target_label(environment_report_str)
            main_logger.info(f"[ManagerAgent] Auto-selected target label for topological enhancement: {self.target_label}")

        return mode

    def decide_target_label(self, environment_report_str) -> int:
        """
        Automatically decide the target label for topology enhancement based on the environment report
        Usually choose the label with the least sample size or uneven distribution
        """
        main_logger = logging.getLogger("main_logger")

        try:
            env_report = json.loads(environment_report_str)
            if "LabelDistribution" not in env_report:
                main_logger.warning("[ManagerAgent] No label distribution in environment report. Defaulting to label 0.")
                return 0

            label_dist = env_report["LabelDistribution"]
            # Find the label with the least number of samples
            min_label = min(label_dist.items(), key=lambda x: int(x[1]))
            target = int(min_label[0]) if min_label[0].isdigit() else 0

            # If no target size is set, it is set to 80% of the average value.
            if self.label_target_size == 0:
                avg_size = sum(int(count) for count in label_dist.values()) / len(label_dist)
                self.label_target_size = int(avg_size * 0.8)

            return target

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            main_logger.error(f"[ManagerAgent] Error deciding target label: {e}. Defaulting to label 0.")
            return 0

    def run_manager_pipeline(self, early_stopping, current_iteration, current_data_str: str = None ) -> (str, bool):
        """
        Perform a single enhancement-evaluation iteration:
            1) Get the latest environment report first
            2) Automatically determine the enhancement mode (semantic or topological) used in this round
            3) Determine sampling (sampled_node_ids)
            4) If current_data_str is not provided, load the initial data
            5) Call the enhancement agent and the evaluation agent
            6) Let LLM decide whether to continue iterating
        """
        main_logger = logging.getLogger("main_logger")  # Get main_logger

        # 1) Get the current environment report (status before this round of enhancement)
        current_environment_report = self.perception_agent.generate_environment_report(
            require_label_distribution=True  # Always obtain label distribution for automatic decision of enhancement mode
        )
        main_logger.info("[ManagerAgent] Generated current environment report for iteration.")

        # 2) Automatically determines the enhancement mode at each turn, unless a fixed mode is specified on the command line.
        if self.enhancement_mode is None or current_iteration % 3 == 0:  # Re-determine every 3 rounds or at the beginning
            prev_mode = self.enhancement_mode
            self.enhancement_mode = self.decide_enhancement_mode(current_environment_report)
            if prev_mode != self.enhancement_mode:
                main_logger.info(f"[ManagerAgent] Enhancement mode changed from {prev_mode} to {self.enhancement_mode}")
            else:
                main_logger.info(f"[ManagerAgent] Enhancement mode remains {self.enhancement_mode}")

        # 3) Determine how to sample based on the current mode:
        if self.enhancement_mode == "semantic":
            self.sampled_node_ids = self.perception_agent.decide_sampling(visualize=self.visualize_sampling)
        else:  # topological mode
            self.sampled_node_ids = self.perception_agent.decide_label_sampling(
                target_label=self.target_label,
                label_target_size=self.label_target_size,
                visualize=self.visualize_sampling
            )

        # 4) If current_data_str is not provided, filter out the required nodes from the file
        if current_data_str is None:
            current_data_str = self.load_initial_data()
            if not current_data_str:
                main_logger.warning("[ManagerAgent] current_data_str is empty. Stopping iteration.")
                return "", False

        # 5) Reinforcement phase
        enhanced_data_str = self.enhancement_agent.enhance_graph(
            data_json_str=current_data_str,
            environment_state_str=current_environment_report,  # Use the current environment report
            mode=self.enhancement_mode,
            target_label=self.target_label,
            label_target_size=self.label_target_size
        )

        if not enhanced_data_str.strip():
            main_logger.error("[ManagerAgent] Enhancement result is empty, stopping iteration.")
            return "", False

        main_logger.info(f'[ManagerAgent] Enhanced data generated successfully.')

        # 6) Assessment phase - incoming initial environmental report, current environmental report and raw data
        evaluated_data_str = self.evaluation_agent.evaluate_graph(
            original_data_str=current_data_str,
            generated_data_str=enhanced_data_str,
            initial_environment_report=self.initial_environment_report,
            current_environment_report=current_environment_report,
            mode=self.enhancement_mode,
            target_label=self.target_label,
            perception_agent=self.perception_agent  # Pass the perception_agent to regenerate the report
        )

        if not evaluated_data_str.strip():
            main_logger.error("[ManagerAgent] Evaluation result is empty, stopping iteration.")
            return "", False

        # 7) Use LLM to determine whether to continue strengthening (if it has reached the early_stopping round, enable LLM decision)
        if current_iteration < early_stopping:
            # Force to continue
            continue_flag = True
        else:
            # Generate a new prompt and let LLM compare the initial and current environment states
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

        return evaluated_data_str, continue_flag