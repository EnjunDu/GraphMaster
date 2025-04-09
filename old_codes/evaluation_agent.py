# evaluation_agent.py

import json
import time
import threading
import tempfile
from transformers import TextGenerationPipeline
import logging


class GraphEvaluationAgent:
    """
    Agent for performing graph evaluation tasks.
    Using the passed TextGenerationPipeline, it evaluates the effect of graph enhancement and determines whether it has converged based on the initial environment report and the current environment report.
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: text generation pipeline (loaded LLM)
        :param max_new_tokens: maximum number of tokens generated each time
        """
        self.text_generation = text_generation_pipeline
        self.max_new_tokens = max_new_tokens

    def evaluate_graph(self,
                       original_data_str: str,
                       generated_data_str: str,
                       initial_environment_report: str,
                       current_environment_report: str,
                       mode: str = "semantic",
                       target_label: str = None,
                       perception_agent = None
                       ) -> str:
        """
        Perform graph evaluation tasks, evaluate the enhancement effect and judge whether it has converged by comparing various environment reports.

        :param original_data_str: Original node data before this round of enhancement (JSON string)
        :param generated_data_str: Enhanced graph data (JSON string)
        :param initial_environment_report: Initial environment report (at the beginning of the project)
        :param current_environment_report: Current environment report (before this round of enhancement)
        :param mode: Enhancement mode ('semantic' or 'topological')
        :param target_label: Target label (when mode='topological')
        :param perception_agent: PerceptionAgent instance, used to generate a new environment report
        :return: Evaluation result (JSON string)
        """
        main_logger = logging.getLogger("main_logger")

        # Merge the original data and the newly generated data, creating a temporary file for PerceptionAgent to analyze
        enhanced_environment_report = ""
        if perception_agent:
            try:
                # Merge data (be careful to handle possible formatting issues)
                original_data = json.loads(original_data_str) if original_data_str else []
                generated_data = json.loads(generated_data_str) if generated_data_str else []

                # If the original data is not a list, try converting it to a list
                if not isinstance(original_data, list):
                    original_data = [original_data]
                if not isinstance(generated_data, list):
                    generated_data = [generated_data]

                # Merge data
                combined_data = original_data + generated_data

                # Write to temporary file
                with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tmp:
                    json.dump(combined_data, tmp, ensure_ascii=False, indent=2)
                    tmp_filename = tmp.name

                # Generate enhanced environment reports using PerceptionAgent
                enhanced_environment_report = perception_agent.generate_environment_report(
                    require_label_distribution=True,
                    data_file=tmp_filename
                )

                main_logger.info(f"[GraphEvaluationAgent] Generated enhanced environment report based on combined data")
            except Exception as e:
                main_logger.error(f"[GraphEvaluationAgent] Error generating enhanced environment report: {e}")
                # If the generation of the enhanced environment report fails, use the current report
                enhanced_environment_report = current_environment_report
        else:
            main_logger.warning("[GraphEvaluationAgent] No perception_agent provided, skipping enhanced report generation")
            enhanced_environment_report = current_environment_report

        if mode == "topological" and target_label:
            prompt_for_evaluation = f"""# Graph Data Evaluation Task: Topological Enhancement for Label = {target_label}

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

        Your task is to:
        1. Compare all environment reports to assess what has changed.
        2. Evaluate if the topological enhancement has improved the representation and connectivity of label={target_label}.
        3. Determine if the enhancement process seems to have converged (minimal changes between iterations).
        4. Evaluate the quality and validity of the newly generated nodes.
        5. Check if the new nodes have appropriate connections to existing nodes.

        If you find any generated data unreasonable, you should remove it.

        When providing the final answer, say "here are the generated datasets:" and then output the JSON data immediately.
        """
        else:  # mode == "semantic"
            prompt_for_evaluation = f"""# Graph Data Evaluation Task: Semantic Enhancement

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

        Evaluation metrics to consider:
        1. Community distribution changes
        2. Node connectivity patterns
        3. Semantic coherence
        4. Label distribution balance
        5. Overall enhancement convergence

        If you find any generated data unreasonable or detrimental to graph quality, you should remove it.

        When providing the final answer, you must say "here are the generated datasets:" and then output the JSON data immediately without extra text.
        """

        main_logger.info(f"[GraphEvaluationAgent] Starting evaluation based on all environment reports comparison, mode={mode}.")

        raw_output = self._call_generation(prompt_for_evaluation, int(self.max_new_tokens * 1.5))
        splitted_flag = "here are the generated datasets:"
        evaluated_json_str = self._extract_after_flag(raw_output, splitted_flag)
        if not evaluated_json_str.strip():
            main_logger.error("[GraphEvaluationAgent] Warning: LLM returned empty evaluation result.")
        return evaluated_json_str

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
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()