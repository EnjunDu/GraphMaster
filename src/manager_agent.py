import json
import os
import time
import threading
import numpy as np
from transformers import TextGenerationPipeline
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent
import logging


class ManagerAgent:
    """
    Manager Agent - serves as the meta-cognitive controller that:
      1) Formulates synthesis queries based on environmental status reports
      2) Orchestrates state transitions in the closed-loop optimization
      3) Decides enhancement modes using multi-objective utility function
      4) Coordinates the perception, enhancement, and evaluation phases
    """

    def __init__(self,
                 text_generation_pipeline: TextGenerationPipeline,
                 perception_agent: GraphPerceptionAgent,
                 enhancement_agent: GraphEnhancementAgent,
                 evaluation_agent: GraphEvaluationAgent,
                 data_file: str,
                 visualize_sampling: bool = False,
                 enhancement_mode: str = None,  # can be None, meaning auto-decide
                 target_label: int = None,
                 label_target_size: int = 0
                 ):
        """
        :param text_generation_pipeline: LLM text generation pipeline
        :param perception_agent: Perception Agent for retrieval
        :param enhancement_agent: Enhancement Agent for generation
        :param evaluation_agent: Evaluation Agent for quality assessment
        :param data_file: JSON file with node information
        :param visualize_sampling: Whether to visualize the sampling process
        :param enhancement_mode: 'semantic' or 'topological' or None (auto-decide)
        :param target_label: Target label for topological enhancement
        :param label_target_size: Desired size for target label
        """
        self.text_generation = text_generation_pipeline

        self.perception_agent = perception_agent
        self.enhancement_agent = enhancement_agent
        self.evaluation_agent = evaluation_agent
        self.data_file = data_file
        self.visualize_sampling = visualize_sampling
        self.enhancement_mode = enhancement_mode  # Can be None
        self.target_label = target_label
        self.label_target_size = label_target_size

        # Initialize adaptive weights for multi-objective utility function
        self.lambda_sem = 0.33  # Weight for semantic coherence
        self.lambda_struct = 0.33  # Weight for structural integrity
        self.lambda_bal = 0.33  # Weight for class balance
        self.eta = 0.05  # Learning rate for weight updates
        
        # Store synthesis progress metrics
        self.synthesis_progress = []
        
        # Store initial environment report (baseline)
        main_logger = logging.getLogger("main_logger")
        try:
            self.initial_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True
            )
            main_logger.info("[ManagerAgent] Initial environment report generated and stored.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating initial environment report: {e}")
            self.initial_environment_report = json.dumps({"error": "Failed to generate initial report"})

        # Auto-decide initial enhancement mode if not specified
        if self.enhancement_mode is None:
            try:
                self.enhancement_mode = self.decide_enhancement_mode(self.initial_environment_report)
                main_logger.info(f"[ManagerAgent] Auto-decided initial enhancement mode: {self.enhancement_mode}")
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error deciding enhancement mode: {e}")
                self.enhancement_mode = "semantic"  # Default to semantic mode
                main_logger.info(f"[ManagerAgent] Falling back to default enhancement mode: {self.enhancement_mode}")

        # No sampling in initialization phase
        self.sampled_node_ids = []

    def load_initial_data(self) -> str:
        """
        Loads data from data_file and filters nodes based on sampled_node_ids.
        Returns JSON string of selected nodes.
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
            main_logger.error("[ManagerAgent] No matching nodes in dataset.")
            return ""
        main_logger.info(f"[ManagerAgent] Found {len(selected_data)} nodes in dataset for enhancement-evaluation.")
        return json.dumps(selected_data, ensure_ascii=False, indent=2)

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
            main_logger.error(f"[ManagerAgent] Error during text generation: {error_dict['error']}")
            return "Error in manager decision. Check logs for details."

        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extracts content after a specified flag in the text.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""

        extracted_json = text[idx + len(flag):].strip()

        # Clean up JSON (remove control characters)
        try:
            extracted_json = extracted_json.replace("\\n", "").replace("\\t", "").replace('\\"', '"')
            json.loads(extracted_json)  # Validate JSON
            return extracted_json
        except json.JSONDecodeError:
            main_logger = logging.getLogger("main_logger")
            main_logger.error("[ManagerAgent] Warning: JSON decoding failed, returning empty string.")
            return ""

    def decide_enhancement_mode(self, environment_report_str) -> str:
        """
        Decides enhancement mode using multi-objective utility function that considers:
        - Semantic coherence
        - Structural integrity
        - Class balance
        
        Returns 'semantic' or 'topological'
        """
        main_logger = logging.getLogger("main_logger")
        main_logger.info("[ManagerAgent] Deciding enhancement mode based on environment report...")

        # Parse environment report
        try:
            env_report = json.loads(environment_report_str)
        except json.JSONDecodeError:
            main_logger.error(
                "[ManagerAgent] Failed to parse environment report JSON. Defaulting to semantic enhancement.")
            return "semantic"

        # Calculate utilities for semantic and topological modes
        semantic_utility = self._calculate_utility("semantic", env_report)
        topological_utility = self._calculate_utility("topological", env_report)
        
        # Choose mode with highest utility
        if semantic_utility > topological_utility:
            mode = "semantic"
        else:
            mode = "topological"
        
        main_logger.info(f"[ManagerAgent] Enhancement mode decision: {mode}")
        main_logger.info(f"[ManagerAgent] Utilities - Semantic: {semantic_utility:.4f}, Topological: {topological_utility:.4f}")
        
        # If topological mode is selected but no target label is specified, auto-decide
        if mode == "topological" and self.target_label is None:
            self.target_label = self.decide_target_label(environment_report_str)
            main_logger.info(f"[ManagerAgent] Auto-selected target label for topological enhancement: {self.target_label}")
        
        return mode

    def _calculate_utility(self, mode, env_report):
        """
        Calculates utility by weighing semantic coherence, structural integrity, and class balance:
        ω* = argmax[λ₁U_sem(ω, G) + λ₂U_struct(ω, G) + λ₃U_bal(ω, G)]
        """
        # Extract metrics from environment report
        if "LabelDistribution" in env_report:
            label_dist = env_report["LabelDistribution"]
            counts = [int(count) for count in label_dist.values()]
            imbalance = max(counts) / (min(counts) + 1e-6)  # Avoid division by zero
        else:
            imbalance = 1.0
        
        # Get community sizes for structural integrity measurement
        community_sizes = []
        if "Communities" in env_report and "sizes" in env_report["Communities"]:
            community_sizes = env_report["Communities"]["sizes"]
            community_size_variance = np.var(community_sizes) if community_sizes else 0
        else:
            community_size_variance = 0
        
        # Calculate individual utilities based on mode
        if mode == "semantic":
            # For semantic mode, higher utility when communities are uniform and classes are balanced
            u_sem = 0.8  # Semantic mode favors semantic coherence
            u_struct = 1.0 / (1.0 + community_size_variance)  # Lower variance is better
            u_bal = 1.0 / (1.0 + imbalance)  # Lower imbalance is better
        else:  # topological mode
            # For topological mode, higher utility when there's class imbalance
            u_sem = 0.5  # Topological mode has moderate semantic coherence
            u_struct = 1.0 / (1.0 + 0.5 * community_size_variance)  # Still care about structure
            u_bal = imbalance / (1.0 + imbalance)  # Higher imbalance means higher utility for topological
        
        # Combine utilities using adaptive weights
        total_utility = (self.lambda_sem * u_sem + 
                         self.lambda_struct * u_struct + 
                         self.lambda_bal * u_bal)
        
        return total_utility

    def decide_target_label(self, environment_report_str) -> int:
        """
        Auto-decides target label for topological enhancement based on class imbalance.
        Usually selects the label with minimum sample count or highest imbalance.
        """
        main_logger = logging.getLogger("main_logger")

        try:
            env_report = json.loads(environment_report_str)
            if "LabelDistribution" not in env_report:
                main_logger.warning(
                    "[ManagerAgent] No label distribution in environment report. Defaulting to label 0.")
                return 0

            label_dist = env_report["LabelDistribution"]
            # Find label with fewest samples
            min_label = min(label_dist.items(), key=lambda x: int(x[1]))
            target = int(min_label[0]) if min_label[0].isdigit() else 0

            # If target size not set, use 80% of average class size
            if self.label_target_size == 0:
                avg_size = sum(int(count) for count in label_dist.values()) / len(label_dist)
                self.label_target_size = int(avg_size * 0.8)

            return target

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            main_logger.error(f"[ManagerAgent] Error deciding target label: {e}. Defaulting to label 0.")
            return 0

    def _update_adaptive_weights(self, current_data, previous_data=None):
        """
        Updates adaptive weights based on synthesis progress:
        λᵢᵗ⁺¹ = λᵢᵗ + η∇ᵢP(G_t)
        """
        if previous_data is None:
            return  # No update if no previous data available
        
        # Calculate synthesis progress as improvement in metrics
        progress = self._measure_synthesis_progress(current_data, previous_data)
        self.synthesis_progress.append(progress)
        
        # Update weights based on progress gradients
        grad_sem = progress.get("semantic_improvement", 0)
        grad_struct = progress.get("structural_improvement", 0)
        grad_bal = progress.get("balance_improvement", 0)
        
        # Update weights
        self.lambda_sem += self.eta * grad_sem
        self.lambda_struct += self.eta * grad_struct
        self.lambda_bal += self.eta * grad_bal
        
        # Normalize weights to sum to 1
        total = self.lambda_sem + self.lambda_struct + self.lambda_bal
        self.lambda_sem /= total
        self.lambda_struct /= total
        self.lambda_bal /= total
        
        main_logger = logging.getLogger("main_logger")
        main_logger.info(f"[ManagerAgent] Updated weights - Semantic: {self.lambda_sem:.2f}, "
                         f"Structural: {self.lambda_struct:.2f}, Balance: {self.lambda_bal:.2f}")

    def _measure_synthesis_progress(self, current_data, previous_data):
        """
        Measures synthesis progress between iterations by analyzing graph metrics.
        Returns improvement gradients for each objective.
        """
        # Use LLM to analyze progress
        prompt = f"""You are analyzing graph synthesis progress.

Previous data:
{previous_data}

Current data:
{current_data}

Calculate improvement metrics:
1. Semantic improvement: How much has text quality and diversity improved? (-1 to 1)
2. Structural improvement: How much has graph connectivity improved? (-1 to 1)
3. Balance improvement: How much has class balance improved? (-1 to 1)

Return a JSON with these three values only.
"""
        try:
            raw_output = self._call_generation(prompt, 512)
            # Extract JSON or numbers
            import re
            sem_match = re.search(r"semantic.*?(-?\d+\.?\d*)", raw_output, re.IGNORECASE)
            struct_match = re.search(r"structural.*?(-?\d+\.?\d*)", raw_output, re.IGNORECASE)
            bal_match = re.search(r"balance.*?(-?\d+\.?\d*)", raw_output, re.IGNORECASE)
            
            semantic_improvement = float(sem_match.group(1)) if sem_match else 0.01
            structural_improvement = float(struct_match.group(1)) if struct_match else 0.01
            balance_improvement = float(bal_match.group(1)) if bal_match else 0.01
        except Exception:
            # Fallback to default values if parsing fails
            semantic_improvement = 0.01
            structural_improvement = 0.01
            balance_improvement = 0.01
            
        return {
            "semantic_improvement": semantic_improvement,
            "structural_improvement": structural_improvement,
            "balance_improvement": balance_improvement
        }

    def run_manager_pipeline(self, early_stopping, current_iteration, current_data_str: str = None) -> (str, bool):
        """
        Executes a single enhancement-evaluation iteration with formal state transitions:
        s_{t+1} = T(s_t, a_t, M_t)
        
        Orchestrates the full RAG cycle:
        1) Query formulation
        2) Knowledge retrieval
        3) Context-conditioned generation
        4) Quality assessment
        """
        main_logger = logging.getLogger("main_logger")

        # Current state s_t includes graph composition and environment
        current_state = {
            "iteration": current_iteration,
            "data": current_data_str,
            "mode": self.enhancement_mode
        }
        
        # Action a_P: Perception - Retrieve relevant context
        # 1) Get current environment report (state before enhancement)
        try:
            current_environment_report = self.perception_agent.generate_environment_report(
                require_label_distribution=True
            )
            main_logger.info("[ManagerAgent] Generated current environment report for iteration.")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error generating current environment report: {e}")
            return "", False

        current_state["environment_report"] = current_environment_report
        
        # 2) Auto-decide enhancement mode every 3 rounds or initially
        if self.enhancement_mode is None or current_iteration % 3 == 0:
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
                    self.enhancement_mode = "semantic"  # Default
                    main_logger.info(f"[ManagerAgent] Setting default enhancement mode: {self.enhancement_mode}")

        # Update state with chosen mode M_t
        current_state["mode"] = self.enhancement_mode
        
        # 3) Use appropriate sampling strategy based on mode
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

        # Update state with sampled nodes (part of knowledge retrieval)
        current_state["sampled_nodes"] = self.sampled_node_ids
        
        # 4) Load initial data if none provided
        if current_data_str is None:
            current_data_str = self.load_initial_data()
            if not current_data_str:
                main_logger.warning("[ManagerAgent] current_data_str is empty. Stopping iteration.")
                return "", False
        
        # Action a_E: Enhancement - Context-conditioned generation
        # 5) Enhancement phase
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

            main_logger.info(f'[ManagerAgent] Enhanced data generated successfully.')
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during enhancement: {e}")
            return "", False

        # Update state with enhancement results
        current_state["enhanced_data"] = enhanced_data_str
        
        # Action a_V: Evaluation - Multi-dimensional quality assessment
        # 6) Evaluation phase
        try:
            evaluated_data_str = self.evaluation_agent.evaluate_graph(
                original_data_str=current_data_str,
                generated_data_str=enhanced_data_str,
                initial_environment_report=self.initial_environment_report,
                current_environment_report=current_environment_report,
                mode=self.enhancement_mode,
                target_label=self.target_label,
                perception_agent=self.perception_agent  # For regenerating reports
            )

            if not evaluated_data_str.strip():
                main_logger.error("[ManagerAgent] Evaluation result is empty, stopping iteration.")
                return "", False
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error during evaluation: {e}")
            return "", False

        # Update state with evaluation results
        current_state["evaluated_data"] = evaluated_data_str
        
        # Update adaptive weights based on synthesis progress
        if current_iteration > 0:
            try:
                self._update_adaptive_weights(evaluated_data_str, current_data_str)
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error updating adaptive weights: {e}")
        
        # 7) Determine if we should continue enhancement
        if current_iteration < early_stopping:
            # Force continuation during initial iterations
            continue_flag = True
        else:
            try:
                # Generate prompt for convergence assessment
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

Please analyze the differences between the initial and current environment reports.
Based on these changes and the evaluation results, determine if we should continue enhancement.

Consider convergence criteria:
1. Quality improvement between iterations has become minimal
2. Synthesis objectives have been achieved
3. Enhancement has reached a stable state

Output True if you see significant potential for further enhancement.
Output False if you detect convergence or minimal improvement.

When providing your decision, say "here is the Final decision:" followed by True or False.
"""

                decision_output = self._call_generation(prompt_manager, max_tokens=4196)
                decision_str = self._extract_after_flag(decision_output, "here is the Final decision:")
                continue_flag = decision_str.lower() == "true"
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error during continuation decision: {e}")
                continue_flag = False

        # Complete state transition
        next_state = {
            "iteration": current_iteration + 1,
            "data": evaluated_data_str,
            "mode": self.enhancement_mode,
            "continue": continue_flag
        }
        
        # Log state transition
        main_logger.info(f"[ManagerAgent] State transition: Iteration {current_iteration} -> {current_iteration + 1}, "
                       f"Continue: {continue_flag}")
        
        return evaluated_data_str, continue_flag