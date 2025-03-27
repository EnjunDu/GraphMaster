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
    The ManagerAgent orchestrates the GraphMaster pipeline:
    1. Dispatches GraphPerceptionAgent to construct the environment report (graph statistics) and decides sampling with LLM aid.
    2. Loads and filters the input dataset based on sampled nodes (initial input for augmentation).
    3. Automatically decides the enhancement mode each round (semantic or topological) if not fixed.
    4. Executes an enhancement-evaluation cycle using the EnhancementAgent and EvaluationAgent.
    5. Returns two outputs for each iteration:
       - evaluated_data_str: the evaluated (possibly cleaned or pruned) graph data as a JSON string.
       - continue_flag: a bool indicating whether further enhancement is beneficial (True) or the process should stop (False).
    """

    def __init__(self,
                 text_generation_pipeline: TextGenerationPipeline,
                 perception_agent: GraphPerceptionAgent,
                 enhancement_agent: GraphEnhancementAgent,
                 evaluation_agent: GraphEvaluationAgent,
                 data_file: str,
                 visualize_sampling: bool = False,
                 enhancement_mode: str = None,
                 target_label: int = None,
                 label_target_size: int = 0):
        """
        Initialize the ManagerAgent with required sub-agents and settings.
        :param text_generation_pipeline: The LLM text generation pipeline (for prompting decisions).
        :param perception_agent: Instance of GraphPerceptionAgent (for environment reporting and sampling).
        :param enhancement_agent: Instance of GraphEnhancementAgent (for data augmentation).
        :param evaluation_agent: Instance of GraphEvaluationAgent (for evaluating the augmented data).
        :param data_file: Path to the original graph data JSON file.
        :param visualize_sampling: Whether to visualize sampling decisions in PerceptionAgent.
        :param enhancement_mode: 'semantic', 'topological', or None (auto-decide each round).
        :param target_label: Target label for augmentation if mode is 'topological'.
        :param label_target_size: Desired total number of nodes of target_label after augmentation.
        """
        self.text_generation = text_generation_pipeline
        self.perception_agent = perception_agent
        self.enhancement_agent = enhancement_agent
        self.evaluation_agent = evaluation_agent
        self.data_file = data_file
        self.visualize_sampling = visualize_sampling
        self.enhancement_mode = enhancement_mode  # If None, will auto-decide
        self.target_label = target_label
        self.label_target_size = label_target_size

        main_logger = logging.getLogger("main_logger")
        # Generate and store the initial environment report (baseline state before any enhancement)
        try:
            self.initial_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True  # always include label distribution to inform mode decision
            )
            main_logger.info("[ManagerAgent] Initial environment report generated and stored.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating initial environment report: {e}")
            self.initial_environment_report = json.dumps({"error": "Failed to generate initial report"})

        # Auto-decide initial enhancement mode if none specified
        if self.enhancement_mode is None:
            try:
                self.enhancement_mode = self.decide_enhancement_mode(self.initial_environment_report)
                main_logger.info(f"[ManagerAgent] Auto-decided initial enhancement mode: {self.enhancement_mode}")
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error deciding enhancement mode: {e}")
                self.enhancement_mode = "semantic"  # default to semantic if decision fails
                main_logger.info(f"[ManagerAgent] Falling back to default enhancement mode: {self.enhancement_mode}")

        # Initially, no nodes are sampled (sampling will occur during pipeline execution)
        self.sampled_node_ids = []

    def load_initial_data(self) -> str:
        """
        Load data from data_file and filter nodes based on sampled_node_ids.
        Returns a JSON string of the selected nodes to be used for augmentation.
        """
        main_logger = logging.getLogger("main_logger")
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
            main_logger.error("[ManagerAgent] No matching nodes in dataset for the sampled IDs.")
            return ""
        main_logger.info(f"[ManagerAgent] Found {len(selected_data)} nodes in dataset for enhancement-evaluation.")
        return json.dumps(selected_data, ensure_ascii=False, indent=2)

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        Internal helper to call the LLM to generate text (used for decision-making prompts).
        Runs generation in a separate thread to avoid blocking, uses greedy decoding for reproducibility.
        """
        result_dict = {}
        error_dict = {}

        def generate_output():
            try:
                output = self.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=False,    # use greedy decoding
                    temperature=0.7,
                    top_p=1
                )
                result_dict["output"] = output[0]["generated_text"]
            except Exception as e:
                error_dict["error"] = str(e)

        main_logger = logging.getLogger("main_logger")
        gen_thread = threading.Thread(target=generate_output)
        gen_thread.start()
        # Wait for generation to complete (checking periodically)
        while gen_thread.is_alive():
            time.sleep(0.5)
        gen_thread.join()

        if "error" in error_dict:
            main_logger.error(f"[ManagerAgent] Error during text generation: {error_dict['error']}")
            return "Error in manager decision. Check logs for details."
        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extract the substring after a flag (e.g., "here is the Final decision:") in the given text.
        This is used to parse the LLM's yes/no or True/False decisions by finding the flag and returning content after it.
        If flag is not found or JSON is malformed, returns an empty string.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        extracted = text[idx + len(flag):].strip()
        # Perform minimal cleanup of extracted text (remove extra quotes or newline artifacts if any)
        extracted = extracted.replace("\\n", "").replace("\\t", "").strip()
        # The decision is expected to be a single word ("True" or "False"), ensure no extra content
        return extracted.split()[0] if extracted else ""

    def decide_enhancement_mode(self, environment_report_str: str) -> str:
        """
        Automatically decide which enhancement mode (semantic or topological) to use based on the environment report.
        Considerations (per the paper's method reasoning):
        1. Community cohesion: if community structure is unclear or weak, lean towards 'topological'.
        2. Label distribution: if labels are imbalanced or scarce, lean towards 'topological'.
        3. Otherwise, or if the graph needs content enrichment, use 'semantic'.
        Returns "semantic" or "topological".
        """
        main_logger = logging.getLogger("main_logger")
        main_logger.info("[ManagerAgent] Deciding enhancement mode based on environment report...")

        try:
            env_report = json.loads(environment_report_str)
        except json.JSONDecodeError:
            main_logger.error("[ManagerAgent] Failed to parse environment report JSON. Defaulting to semantic enhancement.")
            return "semantic"

        # Use LLM to analyze the report and decide the mode (this prompt encapsulates the multi-criteria decision).
        prompt = f"""You are a Graph Analysis Expert.
Based on the following environment report of a graph:
{environment_report_str}

Decide which enhancement mode is more beneficial:
1. "semantic" enhancement – improve node content and attributes.
2. "topological" enhancement – improve graph structure and balance labels.

Factors:
- Clarity of community structure (unclear -> topological).
- Label distribution balance (imbalanced -> topological).
- Node connectivity patterns.
- Current overall graph health.

Output ONLY "semantic" or "topological"."""
        decision = self._call_generation(prompt, max_tokens=100).strip().lower()
        mode = "topological" if "topological" in decision else "semantic"
        main_logger.info(f"[ManagerAgent] Enhancement mode decision: {mode}")

        # If topological is chosen but no specific target label provided, auto-select the label to augment.
        if mode == "topological" and self.target_label is None:
            self.target_label = self.decide_target_label(environment_report_str)
            main_logger.info(f"[ManagerAgent] Auto-selected target label for topological enhancement: {self.target_label}")

        return mode

    def decide_target_label(self, environment_report_str: str) -> int:
        """
        Automatically decide which label to target for topological enhancement, based on the environment report.
        Usually chooses the label with the smallest sample size or highest imbalance.
        """
        main_logger = logging.getLogger("main_logger")
        try:
            env_report = json.loads(environment_report_str)
            # Expecting 'LabelDistribution' in the report for label counts
            if "LabelDistribution" not in env_report:
                main_logger.warning("[ManagerAgent] No label distribution in environment report. Defaulting to label 0.")
                return 0

            label_dist = env_report["LabelDistribution"]
            # Choose the label with the smallest count
            min_label, min_count = None, float("inf")
            for label, count in label_dist.items():
                try:
                    count_val = int(count)
                except (TypeError, ValueError):
                    continue
                if count_val < min_count:
                    min_count = count_val
                    min_label = label
            target = int(min_label) if min_label is not None and str(min_label).isdigit() else 0

            # If no target size specified, set a default target size (80% of average label count)
            if self.label_target_size == 0 and label_dist:
                avg_size = sum(int(c) for c in label_dist.values() if str(c).isdigit()) / len(label_dist)
                self.label_target_size = int(avg_size * 0.8)

            return target
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error deciding target label: {e}. Defaulting to label 0.")
            return 0

    def run_manager_pipeline(self, early_stopping: int, current_iteration: int, current_data_str: str = None) -> (str, bool):
        """
        Execute one enhancement-evaluation iteration:
          1) Generate the current environment report (state before this enhancement).
          2) Decide which enhancement mode to use this round (if not fixed or periodically re-evaluating every 3 rounds).
          3) Perform node sampling based on the selected mode (semantic or topological).
          4) If current_data_str (pre-augmentation nodes) is not provided, load initial data from file for sampled nodes.
          5) Call EnhancementAgent to augment the graph data.
          6) Call EvaluationAgent to evaluate the augmented data against current and initial environment reports.
          7) Use an LLM prompt to decide if further enhancement is needed (considering convergence or improvement).
        Returns:
          evaluated_data_str (JSON string of the evaluated graph data for this iteration),
          continue_flag (bool indicating whether to continue with another iteration).
        """
        main_logger = logging.getLogger("main_logger")
        # Step 1: Get the environment report before enhancement (current state)
        try:
            current_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True
            )
            main_logger.info("[ManagerAgent] Generated current environment report for iteration.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating current environment report: {e}")
            return "", False

        # Step 2: Decide enhancement mode for this iteration (auto-decide unless a fixed mode is set).
        if self.enhancement_mode is None or current_iteration % 3 == 0:
            try:
                prev_mode = self.enhancement_mode
                self.enhancement_mode = self.decide_enhancement_mode(current_environment_report)
                if prev_mode and prev_mode != self.enhancement_mode:
                    main_logger.info(f"[ManagerAgent] Enhancement mode changed from {prev_mode} to {self.enhancement_mode}")
                else:
                    main_logger.info(f"[ManagerAgent] Enhancement mode remains {self.enhancement_mode}")
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error deciding enhancement mode: {e}")
                if self.enhancement_mode is None:
                    self.enhancement_mode = "semantic"
                    main_logger.info(f"[ManagerAgent] Setting default enhancement mode: {self.enhancement_mode}")

        # Step 3: Sampling based on current mode
        try:
            if self.enhancement_mode == "semantic":
                # Semantic mode: sample a smaller community for content enhancement
                self.sampled_node_ids = self.perception_agent.decide_sampling(visualize=self.visualize_sampling)
            else:  # topological mode
                # Topological mode: sample nodes of the target label to augment structure and balance
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

        # Step 4: Prepare the data for enhancement (if not given, load from initial dataset)
        if current_data_str is None:
            current_data_str = self.load_initial_data()
            if not current_data_str:
                main_logger.warning("[ManagerAgent] current_data_str is empty. Stopping iteration.")
                return "", False

        # Step 5: Enhancement stage
        try:
            enhanced_data_str = self.enhancement_agent.enhance_graph(
                data_json_str=current_data_str,
                environment_state_str=current_environment_report,
                mode=self.enhancement_mode,
                target_label=self.target_label,
                label_target_size=self.label_target_size
            )
            if not enhanced_data_str.strip():
                main_logger.error("[ManagerAgent] Enhancement result is empty, stopping iteration.")
                return "", False
            main_logger.info("[ManagerAgent] Enhanced data generated successfully.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during enhancement: {e}")
            return "", False

        # Step 6: Evaluation stage
        try:
            evaluated_data_str = self.evaluation_agent.evaluate_graph(
                original_data_str=current_data_str,
                generated_data_str=enhanced_data_str,
                initial_environment_report=self.initial_environment_report,
                current_environment_report=current_environment_report,
                mode=self.enhancement_mode,
                target_label=self.target_label,
                perception_agent=self.perception_agent  # pass agent to generate a new env report on combined data
            )
            if not evaluated_data_str.strip():
                main_logger.error("[ManagerAgent] Evaluation result is empty, stopping iteration.")
                return "", False
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during evaluation: {e}")
            return "", False

        # Step 7: Decide via LLM if we should continue another iteration (only after reaching early_stopping rounds).
        if current_iteration < early_stopping:
            # Before reaching early_stopping threshold, we always continue (to gather enough data for the LLM decision)
            continue_flag = True
        else:
            try:
                # Build a prompt for the LLM to compare initial vs. current environment state and evaluation outcomes.
                if self.enhancement_mode == "topological":
                    mode_description = f"We have been performing *topological augmentation* focused on label {self.target_label}."
                else:
                    mode_description = "We have been performing *semantic augmentation* on the graph."
                prompt_manager = f"""You are the Manager Agent overseeing graph enhancement.
{mode_description}

## Initial environment report (baseline before any enhancement):
{self.initial_environment_report}

## Environment report before this iteration (current state prior to enhancement):
{current_environment_report}

## Newly generated data in this iteration:
{enhanced_data_str}

## Evaluation result of this iteration:
{evaluated_data_str}

Carefully analyze the differences between the initial and current environment reports and consider the evaluation results.
Determine if further enhancement iterations would yield significant improvements or if the process has converged.

If there is clear potential for further improvement, output **True**.
If the graph has converged or improvements are minimal, output **False**.

When providing your final answer, you must say "here is the Final decision:" and then output **True** or **False** without any extra commentary.
"""
                decision_output = self._call_generation(prompt_manager, max_tokens=2048)
                decision_str = self._extract_after_flag(decision_output, "here is the Final decision:")
                continue_flag = True if decision_str.lower() == "true" else False
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error during continuation decision: {e}")
                # If decision fails, default to stopping further enhancement to avoid infinite loop
                continue_flag = False

        return evaluated_data_str, continue_flag
