# evaluation_agent.py

import json
import time
import threading
import tempfile
import re
from transformers import TextGenerationPipeline
import logging


class GraphEvaluationAgent:
    """
    Evaluation Agent - implements the comprehensive verification mechanism that:
    - Assesses both semantic coherence and structural integrity
    - Computes composite quality scores for generated nodes
    - Uses adaptive thresholding for quality control
    - Determines convergence using temporal quality gradient analysis
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: LLM text generation pipeline
        :param max_new_tokens: Maximum new tokens for generation
        """
        self.text_generation = text_generation_pipeline
        self.max_new_tokens = max_new_tokens
        self.quality_threshold = 7.0  # Initial quality threshold
        self.previous_avg_quality = None  # Store previous average quality
        self.quality_history = []  # Store quality history for convergence detection

    def evaluate_graph(self,
                      original_data_str: str,
                      generated_data_str: str,
                      initial_environment_report: str,
                      current_environment_report: str,
                      mode: str = "semantic",
                      target_label: str = None,
                      perception_agent=None
                      ) -> str:
        """
        Implements a comprehensive verification mechanism that integrates four critical 
        information sources: R_0 (initial environment), R_t (current environment),
        K_t (retrieved knowledge), and G_s^t (newly synthesized data).
        
        Computes composite quality scores and applies adaptive thresholding.
        
        :param original_data_str: Original node data for this round (JSON string)
        :param generated_data_str: Enhanced graph data (JSON string)
        :param initial_environment_report: Initial environment report (project start)
        :param current_environment_report: Current environment report (before this round)
        :param mode: Enhancement mode ('semantic' or 'topological')
        :param target_label: Target label for topological mode
        :param perception_agent: PerceptionAgent instance for generating new reports
        :return: Evaluation result (JSON string)
        """
        main_logger = logging.getLogger("main_logger")

        # Generate enhanced environment report based on combined data
        enhanced_environment_report = ""
        if perception_agent:
            try:
                # Parse and combine data
                original_data = json.loads(original_data_str) if original_data_str else []
                generated_data = json.loads(generated_data_str) if generated_data_str else []

                # Ensure data are lists
                if not isinstance(original_data, list):
                    original_data = [original_data]
                if not isinstance(generated_data, list):
                    generated_data = [generated_data]

                # Combine data
                combined_data = original_data + generated_data

                # Write to temporary file
                with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tmp:
                    json.dump(combined_data, tmp, ensure_ascii=False, indent=2)
                    tmp_filename = tmp.name

                # Generate enhanced environment report
                enhanced_environment_report = perception_agent.generate_environment_report(
                    require_label_distribution=True,
                    data_file=tmp_filename
                )

                main_logger.info(f"[GraphEvaluationAgent] Generated enhanced environment report based on combined data")
            except Exception as e:
                main_logger.error(f"[GraphEvaluationAgent] Error generating enhanced environment report: {e}")
                # Use current report if generation fails
                enhanced_environment_report = current_environment_report
        else:
            main_logger.warning(
                "[GraphEvaluationAgent] No perception_agent provided, skipping enhanced report generation")
            enhanced_environment_report = current_environment_report

        # Create a prompt for evaluation
        prompt_for_evaluation = self._create_evaluation_prompt(
            original_data_str, 
            generated_data_str,
            initial_environment_report,
            current_environment_report,
            enhanced_environment_report,
            mode,
            target_label
        )

        main_logger.info(
            f"[GraphEvaluationAgent] Starting evaluation based on all environment reports comparison, mode={mode}.")

        # Send prompt to LLM
        raw_output = self._call_generation(prompt_for_evaluation, int(self.max_new_tokens * 1.5))
        
        # Extract evaluated JSON and quality scores
        splitted_flag = "here are the generated datasets:"
        evaluated_json_str = self._extract_after_flag(raw_output, splitted_flag)
        
        # Extract quality scores if present
        quality_scores = self._extract_quality_scores(raw_output)
        if quality_scores:
            main_logger.info(f"[GraphEvaluationAgent] Quality scores extracted: {quality_scores}")
        
        # Update adaptive threshold
        avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        self._update_threshold(avg_quality)
        
        # Store quality history for convergence detection
        self.quality_history.append(avg_quality)
        
        # Check for convergence
        convergence_status = self._check_convergence(
            initial_environment_report, enhanced_environment_report, quality_scores
        )
        main_logger.info(f"[GraphEvaluationAgent] Convergence status: {convergence_status}")
        
        if not evaluated_json_str.strip():
            main_logger.error("[GraphEvaluationAgent] Warning: LLM returned empty evaluation result.")
        
        return evaluated_json_str

    def _create_evaluation_prompt(self, original_data_str, generated_data_str, 
                                  initial_environment_report, current_environment_report,
                                  enhanced_environment_report, mode, target_label):
        """
        Creates a comprehensive evaluation prompt that implements the verification mechanism.
        """
        if mode == "topological" and target_label:
            prompt = f"""# Graph Data Evaluation Task: Topological Enhancement for Label = {target_label}

You are a Graph Data Evaluation Agent. We are focusing on evaluating the topological enhancement results for label={target_label}.

## Initial environment report (project start):
{initial_environment_report}

## Current environment report (before this round):
{current_environment_report}

## Enhanced environment report (after this round):
{enhanced_environment_report}

## Original data we started with in this round:
{original_data_str}

## Generated data to evaluate:
{generated_data_str}

Your task is to comprehensively evaluate the quality and effectiveness of the topological enhancement:

1. Compare all environment reports to assess what has changed.
2. Evaluate if the topological enhancement has improved the representation and connectivity of label={target_label}.
3. Determine if the enhancement process seems to have converged (minimal changes between iterations).
4. Evaluate the quality and validity of the newly generated nodes.
5. Check if the new nodes have appropriate connections to existing nodes.

For each generated node, compute a composite quality score (0-10) using these dimensions:
- Semantic coherence (0-10): How well the text matches the label and integrates with existing nodes
- Structural integrity (0-10): How appropriately the node connects with the graph
- Class enhancement value (0-10): How much the node contributes to improving the representation of label={target_label}

Only keep nodes with quality scores above threshold ({self.quality_threshold}).

If you find any generated data unreasonable, you should remove it.

When providing the final answer, include:
1. First say "Quality Scores:" and provide scores for each node
2. Then say "here are the generated datasets:" and output the JSON data immediately.
"""
        else:  # mode == "semantic"
            prompt = f"""# Graph Data Evaluation Task: Semantic Enhancement

You are a Graph Data Evaluation Agent.
Your task is to evaluate the quality of semantic enhancement by comparing environment reports.

## Initial environment report (project start):
{initial_environment_report}

## Current environment report (before this round):
{current_environment_report}

## Enhanced environment report (after this round):
{enhanced_environment_report}

## Original data we started with in this round:
{original_data_str}

## Generated data to evaluate:
{generated_data_str}

Your evaluation should focus on:
1. Changes in graph structure and community distribution across all environment reports
2. Improvements in overall data diversity and semantic richness
3. Signs of convergence (minimal changes between states indicating enhancement has reached its limit)
4. Quality and validity of the newly generated nodes' textual content
5. Semantic consistency of new nodes with their assigned labels

For each generated node, compute a composite quality score (0-10) considering:
- Semantic coherence (0-10): How well the text matches the label and integrates with existing nodes
- Structural integrity (0-10): How appropriately the node connects to the graph
- Overall contribution (0-10): How much the node enhances graph quality and diversity

Only keep nodes with quality scores above threshold ({self.quality_threshold}).

If you find any generated data unreasonable or detrimental to graph quality, you should remove it.

When providing the final answer, include:
1. First say "Quality Scores:" and provide scores for each node
2. Then say "here are the generated datasets:" and output the JSON data immediately.
"""
        return prompt

    def _extract_quality_scores(self, raw_output):
        """
        Extracts quality scores from the raw LLM output.
        """
        quality_scores = {}
        scores_section = ""
        
        # Find the quality scores section
        quality_idx = raw_output.find("Quality Scores:")
        datasets_idx = raw_output.lower().find("here are the generated datasets:")
        
        if quality_idx != -1 and datasets_idx != -1 and quality_idx < datasets_idx:
            scores_section = raw_output[quality_idx:datasets_idx].strip()
            
            # Parse scores - looking for patterns like "new_node 1: 8.5" or similar
            pattern = r"(new_node\s*\d+|node\s*\d+):\s*(\d+\.?\d*)"
            matches = re.findall(pattern, scores_section)
            
            for node_id, score in matches:
                quality_scores[node_id.strip()] = float(score)
        
        return quality_scores

    def _update_threshold(self, avg_quality, zeta=0.1):
        """
        Updates the adaptive threshold based on average quality scores.
        τ_t = τ_(t-1) + ζ(F̄_t - F̄_(t-1))
        """
        # If we have previous average, update threshold
        if self.previous_avg_quality is not None:
            # Calculate new threshold
            self.quality_threshold += zeta * (avg_quality - self.previous_avg_quality)
            # Ensure threshold stays in reasonable range
            self.quality_threshold = max(5.0, min(9.0, self.quality_threshold))
        
        # Update previous average for next iteration
        self.previous_avg_quality = avg_quality
        
        main_logger = logging.getLogger("main_logger")
        main_logger.info(f"[GraphEvaluationAgent] Quality threshold updated to: {self.quality_threshold:.2f}")
        
        return self.quality_threshold

    def _check_convergence(self, initial_report, current_report, quality_scores, epsilon=0.05, window_size=3):
        """
        Determines convergence using temporal quality gradient analysis:
        Converged_t = I(max_{j ∈ {1,...,k}} |F̄_t - F̄_(t-j)| < ε ∧ LLM_goal(R_0, R_t) = True)
        """
        main_logger = logging.getLogger("main_logger")
        
        # Check if quality scores are available
        if not quality_scores:
            return {"converged": False, "reason": "No quality scores available"}
        
        # Calculate average quality
        avg_quality = sum(quality_scores.values()) / len(quality_scores)
        
        # Temporal quality gradient analysis
        gradient_converged = False
        if len(self.quality_history) >= window_size:
            # Check if quality improvement has plateaued
            max_gradient = max([abs(avg_quality - self.quality_history[-j-1]) 
                                for j in range(min(window_size, len(self.quality_history)-1))])
            gradient_converged = max_gradient < epsilon
        
        # Use LLM to check if synthesis objectives have been achieved
        goal_achievement = self._assess_goal_achievement(initial_report, current_report)
        
        # Determine overall convergence
        converged = gradient_converged and goal_achievement.get("achieved", False)
        
        return {
            "converged": converged,
            "avg_quality": avg_quality,
            "gradient_converged": gradient_converged,
            "goal_achieved": goal_achievement.get("achieved", False),
            "reason": goal_achievement.get("reason", "Unknown")
        }

    def _assess_goal_achievement(self, initial_report, current_report):
        """
        Uses LLM to assess whether synthesis objectives have been achieved.
        """
        prompt = f"""You are a Graph Data Evaluation Agent assessing whether synthesis objectives have been achieved.

## Initial environment report (project start):
{initial_report}

## Current environment report (after enhancements):
{current_report}

Based on these reports, determine if the graph enhancement objectives have been achieved.
Consider these criteria:
1. Improvement in graph structure and connectivity
2. Better balance in label distribution
3. Enhanced semantic coherence
4. Sufficient data volume for all labels/classes

Output your assessment as a JSON with two fields:
1. "achieved": true or false
2. "reason": brief explanation of your decision

Your response should be ONLY the JSON with no additional text.
"""
        
        # Call LLM with shorter max tokens
        output = self._call_generation(prompt, max_tokens=512)
        
        # Try to extract JSON
        try:
            # Find JSON-like content in the output
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, output, re.DOTALL)
            if match:
                json_str = match.group(0)
                result = json.loads(json_str)
                return result
        except Exception as e:
            main_logger = logging.getLogger("main_logger")
            main_logger.error(f"[GraphEvaluationAgent] Error parsing goal achievement: {e}")
        
        # Default return if parsing fails
        return {"achieved": False, "reason": "Failed to assess goal achievement"}

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
            main_logger.error(f"[GraphEvaluationAgent] Error during text generation: {error_dict['error']}")
            return "Error generating evaluation. Check logs for details."

        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extracts content after a specified flag in the text.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()