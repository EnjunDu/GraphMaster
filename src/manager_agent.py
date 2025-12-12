import json
import logging
import os
import tempfile
import re
from typing import Tuple
from transformers import TextGenerationPipeline
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent


def save_enhanced_data_to_file(enhanced_file_path, enhanced_json_str):
    main_logger = logging.getLogger("main_logger")
    
    if not enhanced_json_str or enhanced_json_str.strip() == "[]":
        main_logger.warning("[save_enhanced_data] No data to save")
        return
    
    try:
        new_nodes = json.loads(enhanced_json_str)
        
        if not isinstance(new_nodes, list):
            main_logger.error(f"[save_enhanced_data] Data is not a list: {type(new_nodes)}")
            return
        
        if len(new_nodes) == 0:
            main_logger.warning("[save_enhanced_data] Empty node list")
            return
        
        main_logger.info(f"[save_enhanced_data] Parsed {len(new_nodes)} new nodes")
        
        existing_data = []
        if os.path.exists(enhanced_file_path):
            try:
                with open(enhanced_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                main_logger.info(f"[save_enhanced_data] Loaded {len(existing_data)} existing nodes")
            except (json.JSONDecodeError, IOError) as e:
                main_logger.error(f"[save_enhanced_data] Error loading existing data: {e}")
                existing_data = []
        else:
            main_logger.info(f"[save_enhanced_data] File does not exist, will create new: {enhanced_file_path}")
        
        existing_ids = {node.get('node_id') for node in existing_data if 'node_id' in node}
        main_logger.info(f"[save_enhanced_data] Found {len(existing_ids)} existing node IDs")
        
        unique_new_nodes = [node for node in new_nodes if node.get('node_id') not in existing_ids]
        
        if len(unique_new_nodes) < len(new_nodes):
            main_logger.warning(
                f"[save_enhanced_data] Filtered out {len(new_nodes) - len(unique_new_nodes)} duplicate nodes"
            )
        
        if len(unique_new_nodes) == 0:
            main_logger.warning("[save_enhanced_data] No new unique nodes to add")
            return
        
        max_id = 0
        for node in existing_data:
            node_id = node.get('node_id', '')
            if isinstance(node_id, int):
                max_id = max(max_id, node_id)
            elif isinstance(node_id, str):
                numbers = re.findall(r'\d+', node_id)
                if numbers:
                    max_id = max(max_id, int(numbers[-1]))
        
        main_logger.info(f"[save_enhanced_data] Max existing node ID: {max_id}")
        
        for i, node in enumerate(unique_new_nodes, start=1):
            old_id = node.get('node_id', '')
            new_id = f"new_node_{max_id + i}"
            node['node_id'] = new_id
            main_logger.debug(f"[save_enhanced_data] Renumbered {old_id} -> {new_id}")
            
            if 'neighbors' in node:
                updated_neighbors = []
                for neighbor_id in node['neighbors']:
                    found = False
                    for j, other_node in enumerate(unique_new_nodes):
                        if other_node.get('node_id') == neighbor_id:
                            updated_neighbors.append(f"new_node_{max_id + j + 1}")
                            found = True
                            break
                    if not found:
                        updated_neighbors.append(neighbor_id)
                node['neighbors'] = updated_neighbors
        
        combined_data = existing_data + unique_new_nodes
        
        with open(enhanced_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        main_logger.info(f"[save_enhanced_data] ✓ Successfully saved enhanced data:")
        main_logger.info(f"[save_enhanced_data]   - Existing nodes: {len(existing_data)}")
        main_logger.info(f"[save_enhanced_data]   - New nodes added: {len(unique_new_nodes)}")
        main_logger.info(f"[save_enhanced_data]   - Total nodes: {len(combined_data)}")
        main_logger.info(f"[save_enhanced_data]   - File: {enhanced_file_path}")
        
    except json.JSONDecodeError as e:
        main_logger.error(f"[save_enhanced_data] JSON parse error: {e}")
        main_logger.error(f"[save_enhanced_data] Problematic data (first 500 chars): {enhanced_json_str[:500]}")
    except Exception as e:
        main_logger.error(f"[save_enhanced_data] Unexpected error: {e}")
        import traceback
        main_logger.error(f"[save_enhanced_data] Traceback:\n{traceback.format_exc()}")


class ManagerAgent:
    """
    Manager Agent - orchestrates the RAG-based enhancement pipeline
    Coordinates retrieval (perception), generation (enhancement), and verification (evaluation)
    """

    def __init__(
        self,
        text_generation_pipeline: TextGenerationPipeline,
        perception_agent: GraphPerceptionAgent,
        enhancement_agent: GraphEnhancementAgent,
        evaluation_agent: GraphEvaluationAgent,
        data_file: str,
        visualize_sampling: bool = False,
        enhancement_mode: str = None,
        target_label: int = None,
        label_target_size: int = 0,
    ):
        self.text_generation = text_generation_pipeline
        self.perception_agent = perception_agent
        self.enhancement_agent = enhancement_agent
        self.evaluation_agent = evaluation_agent
        self.data_file = data_file
        self.visualize_sampling = visualize_sampling
        
        self.enhancement_mode = enhancement_mode  # None means automatic decision
        self.target_label = target_label
        self.label_target_size = label_target_size
        
        self.lambda_sem = 0.4
        self.lambda_struct = 0.4
        self.lambda_bal = 0.2
        
        self.iteration_history = []
        
        self.initial_environment_report = None
        
    def run_manager_pipeline(
        self, 
        early_stopping: int = 10, 
        current_iteration: int = 0,
        current_data_str: str = None
    ) -> Tuple[str, bool]:
        main_logger = logging.getLogger("main_logger")
        
        main_logger.info("[ManagerAgent] Step 1: Generating environment report (RAG Retrieval)")
        
        environment_report = self.perception_agent.generate_environment_report(
            require_label_distribution=True
        )
        main_logger.info("[ManagerAgent] Generated current environment report for iteration.")
        
        if self.initial_environment_report is None:
            main_logger.info("[ManagerAgent] Generating initial environment report for first iteration")
            self.initial_environment_report = environment_report
        
        if self.enhancement_mode is None:
            main_logger.info("[ManagerAgent] Deciding enhancement mode based on environment report...")
            decided_mode = self._decide_enhancement_mode(environment_report)
            main_logger.info(f"[ManagerAgent] Enhancement mode decision: {decided_mode}")
        else:
            decided_mode = self.enhancement_mode
            main_logger.info(f"[ManagerAgent] Using fixed enhancement mode: {decided_mode}")
        
        previous_mode = self.enhancement_mode if hasattr(self, 'enhancement_mode') else None
        self.enhancement_mode = decided_mode
        
        if previous_mode and previous_mode != decided_mode:
            main_logger.info(f"[ManagerAgent] Enhancement mode switched: {previous_mode} -> {decided_mode}")
        
        main_logger.info("[ManagerAgent] Step 2: Sampling data for enhancement (RAG Retrieval)")
        
        if self.enhancement_mode == "topological" and self.target_label is not None:
            main_logger.info(f"[ManagerAgent] Topological mode: sampling nodes with label={self.target_label}")
            
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                
                target_nodes = [node for node in all_data if node.get('label') == self.target_label]
                
                main_logger.info(f"[ManagerAgent] Found {len(target_nodes)} nodes with label={self.target_label}")
                
                if len(target_nodes) == 0:
                    main_logger.error(f"[ManagerAgent] No nodes found with label={self.target_label}")
                    return "", False
                
                sample_size = min(len(target_nodes), 30)
                
                import random
                if len(target_nodes) > sample_size:
                    sampled_nodes = random.sample(target_nodes, sample_size)
                else:
                    sampled_nodes = target_nodes
                
                sampled_data_json_str = json.dumps(sampled_nodes, ensure_ascii=False, indent=2)
                
                main_logger.info(f"[ManagerAgent] Sampled {len(sampled_nodes)} nodes for topological enhancement")
                
            except Exception as e:
                main_logger.error(f"[ManagerAgent] Error sampling for topological mode: {e}")
                import traceback
                main_logger.error(f"[ManagerAgent] Traceback:\n{traceback.format_exc()}")
                return "", False
        else:
            main_logger.info("[ManagerAgent] Semantic mode: using PPR-based sampling")
            sampled_data_json_str = self.perception_agent.sample_high_ppr_nodes(
                visualize=self.visualize_sampling
            )
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                original_dataset = json.load(f)
            main_logger.info(f"[ManagerAgent] Loaded dataset from {self.data_file}, total nodes: {len(original_dataset)}")
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error loading dataset: {e}")
            return "", False
        
        try:
            sampled_data = json.loads(sampled_data_json_str)
            main_logger.info(f"[ManagerAgent] Found {len(sampled_data)} nodes in dataset for enhancement-evaluation.")
        except json.JSONDecodeError as e:
            main_logger.error(f"[ManagerAgent] Error parsing sampled data: {e}")
            return "", False
        
        main_logger.info("[ManagerAgent] Step 3: Graph enhancement (RAG Generation)")
        
        enhanced_json_str = self.enhancement_agent.enhance_graph(
            data_json_str=sampled_data_json_str,
            environment_state_str=environment_report,
            mode=self.enhancement_mode,
            target_label=self.target_label,
            label_target_size=self.label_target_size
        )
        
        if not enhanced_json_str or enhanced_json_str.strip() == "[]":
            main_logger.warning("[ManagerAgent] No enhanced data generated.")
            return "", False
        
        main_logger.info("[ManagerAgent] Enhanced data generated successfully.")
        
        main_logger.info("[ManagerAgent] Step 4: Evaluating enhanced data (RAG Verification)")
        
        evaluated_data_str, continue_flag = self.evaluation_agent.evaluate_graph(
            original_data_str=sampled_data_json_str,
            generated_data_str=enhanced_json_str,
            initial_environment_report=self.initial_environment_report,
            current_environment_report=environment_report,
            mode=self.enhancement_mode,
            target_label=self.target_label,
            perception_agent=self.perception_agent,
            early_stopping=early_stopping
        )
        
        main_logger.info(f"[ManagerAgent] Evaluation completed, continue_flag={continue_flag}")
        
        self._update_adaptive_weights(environment_report, self.enhancement_mode)
        
        enhanced_file = self.data_file
        if not enhanced_file.endswith('_enhanced.json'):
            enhanced_file = enhanced_file.replace('.json', '_enhanced.json')
        
        main_logger.info(f"[ManagerAgent] Step 5: Saving enhanced data to {enhanced_file}")
        save_enhanced_data_to_file(enhanced_file, evaluated_data_str)
        
        iteration_record = {
            "iteration": current_iteration + 1,
            "mode": self.enhancement_mode,
            "weights": {
                "semantic": self.lambda_sem,
                "structural": self.lambda_struct,
                "balance": self.lambda_bal
            },
            "continue_flag": continue_flag
        }
        self.iteration_history.append(iteration_record)
        
        main_logger.info(f"[ManagerAgent] Iteration {current_iteration + 1} completed")
        main_logger.info(f"[ManagerAgent] Continue flag: {continue_flag}")
        
        return evaluated_data_str, continue_flag
    
    def _decide_enhancement_mode(self, environment_report: str) -> str:
        main_logger = logging.getLogger("main_logger")
        
        try:
            env_data = json.loads(environment_report)
            
            label_dist = env_data.get("LabelDistribution", {})
            if not label_dist:
                main_logger.warning("[ManagerAgent] No label distribution found, defaulting to semantic mode")
                return "semantic"
            
            label_counts = list(label_dist.values())
            max_count = max(label_counts)
            min_count = min(label_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            avg_degree = env_data.get("AverageDegree", 0)
            
            if imbalance_ratio > 3.0:
                main_logger.info(f"[ManagerAgent] High imbalance detected (ratio={imbalance_ratio:.2f}), choosing topological mode")
                
                min_label = min(label_dist.items(), key=lambda x: x[1])
                self.target_label = int(min_label[0])
                self.label_target_size = max_count
                
                main_logger.info(f"[ManagerAgent] Auto-set target_label={self.target_label}, target_size={self.label_target_size}")
                
                return "topological"
            else:
                main_logger.info(f"[ManagerAgent] Balanced distribution (ratio={imbalance_ratio:.2f}), choosing semantic mode")
                return "semantic"
            
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error in mode decision: {e}")
            main_logger.info("[ManagerAgent] Defaulting to semantic mode")
            return "semantic"
    
    def _update_adaptive_weights(self, environment_report: str, mode: str):
        main_logger = logging.getLogger("main_logger")
        
        try:
            env_data = json.loads(environment_report)
            
            if mode == "semantic":
                self.lambda_sem = 0.5
                self.lambda_struct = 0.3
                self.lambda_bal = 0.2
            else:
                self.lambda_sem = 0.2
                self.lambda_struct = 0.4
                self.lambda_bal = 0.4
            
            main_logger.info(
                f"[ManagerAgent] Updated adaptive weights for {mode} mode: "
                f"semantic={self.lambda_sem:.2f}, structural={self.lambda_struct:.2f}, balance={self.lambda_bal:.2f}"
            )
            
        except Exception as e:
            main_logger.error(f"[ManagerAgent] Error updating weights: {e}")