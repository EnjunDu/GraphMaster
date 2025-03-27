import json
import time
import threading
import tempfile
from transformers import TextGenerationPipeline
import logging

class GraphEvaluationAgent:
    """
    Agent for evaluating the results of graph enhancement.
    It uses the initial and current environment reports to assess improvements and convergence, and it ensures the newly generated data is reasonable.
    The agent may instruct removal or modification of unreasonable new nodes via the LLM, effectively 'cleaning' the augmented data.
    """

    def __init__(self, text_generation_pipeline: TextGenerationPipeline, max_new_tokens: int = 1024):
        """
        :param text_generation_pipeline: LLM text generation pipeline (pre-loaded model).
        :param max_new_tokens: Maximum tokens to generate for evaluation descriptions.
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
                       perception_agent=None) -> str:
        """
        Evaluate the augmented graph data by comparing environment reports.
        Merges original and generated data to form an enhanced graph, gets an enhanced environment report (if perception_agent provided),
        then uses the LLM to analyze differences and quality.
        :param original_data_str: JSON string of the graph data before this enhancement (the input to augmentation).
        :param generated_data_str: JSON string of the newly generated graph data (augmentation output).
        :param initial_environment_report: JSON string of the initial environment report (baseline at start of pipeline).
        :param current_environment_report: JSON string of the environment report before this enhancement (state prior to augmentation).
        :param mode: Enhancement mode ('semantic' or 'topological') used this round.
        :param target_label: Target label (if mode='topological').
        :param perception_agent: Instance of GraphPerceptionAgent to generate an environment report for the combined data.
        :return: Evaluated result as a JSON string (likely a list of final nodes to use after cleaning).
        """
        main_logger = logging.getLogger("main_logger")

        # Combine original and generated data into one dataset to analyze the new state
        enhanced_environment_report = ""
        if perception_agent:
            try:
                original_data = json.loads(original_data_str) if original_data_str else []
                generated_data = json.loads(generated_data_str) if generated_data_str else []
                if not isinstance(original_data, list):
                    original_data = [original_data]
                if not isinstance(generated_data, list):
                    generated_data = [generated_data]
                combined_data = original_data + generated_data
                # Write combined data to a temporary file to generate its environment report
                with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as tmp:
                    json.dump(combined_data, tmp, ensure_ascii=False, indent=2)
                    tmp_filename = tmp.name
                # Use perception_agent to generate environment report for the enhanced graph state
                enhanced_environment_report = perception_agent.generate_environment_report(
                    require_label_distribution=True,
                    data_file=tmp_filename
                )
                main_logger.info("[GraphEvaluationAgent] Generated enhanced environment report for combined data.")
            except Exception as e:
                main_logger.error(f"[GraphEvaluationAgent] Error generating enhanced environment report: {e}")
                # If failing to generate, fallback to using the current_environment_report as a proxy
                enhanced_environment_report = current_environment_report
        else:
            main_logger.warning("[GraphEvaluationAgent] No perception_agent provided, skipping enhanced environment report generation.")
            enhanced_environment_report = current_environment_report

        # Build the evaluation prompt for the LLM
        if mode == "topological" and target_label is not None:
            prompt_for_evaluation = f"""# Graph Data Evaluation Task: Topological Enhancement (Label {target_label})

You are a Graph Evaluation Agent reviewing a topological augmentation focused on label {target_label}.

## Initial environment report (baseline at start):
{initial_environment_report}

## Environment report before this iteration (pre-enhancement):
{current_environment_report}

## Environment report after this iteration (post-enhancement):
{enhanced_environment_report}

## Original data before enhancement (this iteration's input):
{original_data_str}

## Generated data from enhancement (new nodes added):
{generated_data_str}

Evaluation instructions:
1. Compare the environment reports to identify structural or distribution changes.
2. Check if the topological enhancement improved representation and connectivity of label {target_label}.
3. Determine if the enhancement process is converging (are changes getting smaller compared to previous state?).
4. Evaluate quality and validity of newly generated nodes and their connections.
5. Remove or mark any generated nodes that seem incorrect or harmful to graph integrity.

Finally, output the final graph data (combined original + modified new data) as JSON.
Begin your answer with "here are the generated datasets:" followed by the JSON.
"""
        else:
            prompt_for_evaluation = f"""# Graph Data Evaluation Task: Semantic Enhancement

You are a Graph Evaluation Agent reviewing a semantic augmentation.

## Initial environment report (baseline at start):
{initial_environment_report}

## Environment report before this iteration (pre-enhancement):
{current_environment_report}

## Environment report after this iteration (post-enhancement):
{enhanced_environment_report}

## Original data before enhancement (this iteration's input):
{original_data_str}

## Generated data from enhancement (new nodes added):
{generated_data_str}

Evaluation focus:
- Highlight changes in graph structure and community distribution across the reports.
- Assess improvements in data diversity and semantic richness due to the new nodes.
- Check if the process is converging (diminishing changes indicating it may be time to stop).
- Verify the quality of new nodes' text and their semantic consistency with labels.
- Ensure new nodes have appropriate neighbor connections and do not violate graph logic.

If any generated node is inappropriate or redundant, suggest its removal.

Finally, output the final updated graph data (original plus any retained new nodes) as JSON.
Begin with "here are the generated datasets:" followed by the JSON.
"""
        main_logger.info(f"[GraphEvaluationAgent] Running evaluation for mode={mode} (target_label={target_label}).")
        raw_output = self._call_generation(prompt_for_evaluation, int(self.max_new_tokens * 1.5))
        splitted_flag = "here are the generated datasets:"
        evaluated_json_str = self._extract_after_flag(raw_output, splitted_flag)
        if not evaluated_json_str.strip():
            main_logger.error("[GraphEvaluationAgent] Evaluation LLM returned empty result; proceeding with combined data as default.")
            # If LLM fails to produce output, just return combined data (the original + generated unmodified)
            try:
                combined = json.loads(original_data_str) + json.loads(generated_data_str)
                evaluated_json_str = json.dumps(combined, ensure_ascii=False, indent=2)
            except Exception:
                evaluated_json_str = original_data_str  # fallback to original data if all else fails
        return evaluated_json_str

    def _call_generation(self, prompt: str, max_tokens: int) -> str:
        """
        Internal helper to invoke the LLM for evaluation.
        Uses a separate thread and greedy decoding to ensure a deterministic outcome if possible.
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
            main_logger.error(f"[GraphEvaluationAgent] Error during text generation: {error_dict['error']}")
            return ""
        return result_dict.get("output", "").strip()

    def _extract_after_flag(self, text: str, flag: str) -> str:
        """
        Extract substring after a given flag (case-insensitive) in the text.
        Used to isolate JSON output after a known phrase.
        """
        idx = text.lower().find(flag.lower())
        if idx == -1:
            return ""
        return text[idx + len(flag):].strip()
