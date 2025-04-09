import os
import json
import argparse
import torch
import re
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from manager_agent import ManagerAgent
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    main_logger, result_logger = setup_logging()
    parser = argparse.ArgumentParser(description="GraphMaster: A RAG-Based Multi-Agent Framework for Text-Attributed Graph Synthesis")
    parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated list of GPU indices to be used")
    parser.add_argument('--cache_dir', type=str, default="../models", help="HF model cache directory")
    parser.add_argument('--max_tokens', type=int, default=9196, help="Max new tokens for generation")
    parser.add_argument('--max_iterations', type=int, default=100, help="Max iteration loops for enhancement-evaluation")
    parser.add_argument('--data_file', type=str, default="../data/SubCora.json", help="Path to the input data file")
    parser.add_argument('--llm_model', type=str, default="QwQ", 
                        help="Available options: QwQ, Deepseek, Qwen (Qwen1.5-32B), LLaMA (Samantha-1.1-llama-33b), or custom path")
    parser.add_argument('--hf_token', type=str, default=None, 
                        help="Hugging Face token for accessing gated models")
    parser.add_argument('--visualize_sampling', action='store_true', help="Whether to visualize the sampling result in PerceptionAgent")
    parser.add_argument('--early_stopping', type= int, default=10)
    parser.add_argument('--enhancement_mode', type=str, default=None,
                        choices=['semantic', 'topological', None],
                        help="Choose enhancement mode: 'semantic' or 'topological' or None (auto-decide by agent)")
    parser.add_argument('--target_label', type=int, default=None,
                        help="Which label to augment if mode is 'topological'")
    parser.add_argument('--label_target_size', type=int, default=0,
                        help="Desired total number of data for that label after augmentation (if mode is 'topological')")
    parser.add_argument('--top_percent', type=float, default=0.1,
                        help="Percentage of nodes to sample from the highest PPR score community")
    parser.add_argument('--sample_size', type=int, default=30,
                        help="Number of nodes to sample from the highest PPR score community")
    args = parser.parse_args()

    # Standardize model name by removing numbers and dashes
    model_key = args.llm_model.lower().replace("-", "").replace("32b", "").replace("33b", "")
    
    # Map standardized keys to actual model names
    model_mapping = {
        "qwq": "Qwen/QwQ-32B-Preview",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwen": "Qwen/Qwen1.5-32B",
        "llama": "cognitivecomputations/samantha-1.1-llama-33b"  # Using Samantha 1.1 LLaMA 33B
    }
    
    if model_key in model_mapping:
        args.model_name = model_mapping[model_key]
        main_logger.info(f"[main] Using model: {args.model_name}")
    else:
        # Check if the input might be a direct model path
        if "/" in args.llm_model:
            args.model_name = args.llm_model
            main_logger.info(f"[main] Using custom model path: {args.model_name}")
        else:
            valid_options = ", ".join(model_mapping.keys())
            main_logger.error(f"[main] Unknown model option: {args.llm_model}. Valid options: {valid_options}")
            main_logger.info("[main] Defaulting to QwQ-32B...")
            args.model_name = model_mapping["qwq"]

    # ========== 1. Set environment variables and devices ==========
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = args.cache_dir

    # Print debug info about available devices
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    main_logger.info(f"[main] Visible devices: {visible_devices}")
    main_logger.info(f"[main] CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        main_logger.info(f"[main] GPU {i}: {torch.cuda.get_device_name(i)}, "
                        f"Free: {torch.cuda.mem_get_info(i)[0]/1024**3:.2f} GiB, "
                        f"Total: {torch.cuda.mem_get_info(i)[1]/1024**3:.2f} GiB")
        
    # ========== 2. Load LLM and pipeline ==========
    ensure_model_downloaded(args.model_name, args.cache_dir)

    main_logger.info("[main] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map="auto",  # Auto-allocate to GPUs
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    text_generation_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    main_logger.info("[main] Model and tokenizer loaded successfully.")

    # ========== 3. Initialize Agents ==========
    main_logger.info("[main] Initializing PerceptionAgent for retrieval component...")
    perception_agent = GraphPerceptionAgent(
        data_file=args.data_file,
        llm_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens,
        top_percent=args.top_percent,
        sample_size=args.sample_size
    )
    
    main_logger.info("[main] Initializing EnhancementAgent for generation component...")
    enhancement_agent = GraphEnhancementAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens
    )
    
    main_logger.info("[main] Initializing EvaluationAgent for verification component...")
    evaluation_agent = GraphEvaluationAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens
    )

    # ========== 4. Initialize ManagerAgent ==========
    main_logger.info(f"[main] Initializing ManagerAgent with enhancement_mode={args.enhancement_mode}...")
    manager_agent = ManagerAgent(
        text_generation_pipeline=text_generation_pipeline,
        perception_agent=perception_agent,
        enhancement_agent=enhancement_agent,
        evaluation_agent=evaluation_agent,
        data_file=args.data_file,
        visualize_sampling=args.visualize_sampling,
        enhancement_mode=args.enhancement_mode,  # Can be None, auto-decide
        target_label=args.target_label,
        label_target_size=args.label_target_size
    )

    # ========== 5. RAG-Based Iterative Enhancement Loop ==========
    main_logger.info("[main] Starting GraphMaster RAG-based iterative enhancement loop...")
    iteration = 0
    continue_flag = True
    iteration_history = []  # Store history for visualization
    current_data_str = None  # Will be loaded by manager_agent
    
    # Create enhanced data file
    original_data_file = args.data_file
    enhanced_data_file = original_data_file.replace(".json", "_enhanced.json")
    if not os.path.exists(enhanced_data_file):
        create_enhanced_file(original_data_file, enhanced_data_file)
        main_logger.info(f"[main] Enhanced file created: {enhanced_data_file} (copied from {original_data_file})")
    else:
        main_logger.info(f"[main] Found existing enhanced file: {enhanced_data_file}. Continuing enhancement.")
        
    # Update data file for enhancement
    args.data_file = enhanced_data_file
    
    # Log baseline state
    main_logger.info("[main] Generating baseline environment report...")
    baseline_report = perception_agent.generate_environment_report(require_label_distribution=True)
    result_logger.info(f"********* BASELINE STATE *********\n{baseline_report}\n")
    
    # Iterative RAG optimization loop
    while continue_flag and iteration < args.max_iterations:
        main_logger.info(f"\n[main] --- Iteration {iteration + 1} ---")
        main_logger.info(f"[main] Current enhancement mode: {manager_agent.enhancement_mode}")
        
        # Execute one iteration of the RAG pipeline
        evaluated_data_str, continue_flag = manager_agent.run_manager_pipeline(
            early_stopping=args.early_stopping, 
            current_iteration=iteration,
            current_data_str=current_data_str
        )
        
        if not evaluated_data_str.strip():
            main_logger.info("[main] No valid enhancement data obtained. Stopping iterations.")
            break

        # Clean and format the JSON data
        cleaned_data_str = reformat_json_str(evaluated_data_str)
        main_logger.info(f"[main] Cleaned JSON data at iteration {iteration + 1}")
        
        # Store current iteration's result
        current_mode = manager_agent.enhancement_mode
        iteration_data = {
            "iteration": iteration + 1, 
            "mode": current_mode,
            "enhanced_data": cleaned_data_str,
            "weights": {
                "semantic": manager_agent.lambda_sem,
                "structural": manager_agent.lambda_struct,
                "balance": manager_agent.lambda_bal
            }
        }
        iteration_history.append(iteration_data)
        
        # Update the enhanced data file
        update_enhanced_data(enhanced_data_file, cleaned_data_str)
        
        # Log the result
        result_logger.info(f"**************** ITERATION {iteration+1} ****************")
        result_logger.info(f"Mode: {current_mode}")
        result_logger.info(f"Adaptive Weights: Semantic={manager_agent.lambda_sem:.2f}, " +
                         f"Structural={manager_agent.lambda_struct:.2f}, " +
                         f"Balance={manager_agent.lambda_bal:.2f}")
        result_logger.info(f"Enhanced Data: {cleaned_data_str}")
        
        # Update current_data_str for next iteration
        current_data_str = cleaned_data_str
        
        # Increment iteration counter
        iteration += 1

        if not continue_flag:
            main_logger.info("[main] ManagerAgent indicates optimization convergence.")
            break
    
    # ========== 6. Generate Final Report and Visualizations ==========
    main_logger.info("[main] Generating final environment report...")
    final_report = perception_agent.generate_environment_report(require_label_distribution=True)
    
    # Visualize the enhancement process
    if args.visualize_sampling:
        generate_enhancement_visualization(iteration_history, baseline_report, final_report)
    
    # Log final state
    result_logger.info(f"********* FINAL STATE AFTER {iteration} ITERATIONS *********")
    result_logger.info(f"Final Enhancement Mode: {manager_agent.enhancement_mode}")
    result_logger.info(f"Final Report: {final_report}")
    
    main_logger.info("[main] GraphMaster RAG-based enhancement pipeline completed successfully.")

def generate_enhancement_visualization(iteration_history, baseline_report, final_report):
    """
    Generates visualizations of the enhancement process to show progress.
    """
    main_logger = logging.getLogger("main_logger")
    main_logger.info("[main] Generating enhancement process visualization...")
    
    try:
        # Extract data for visualization
        iterations = [data["iteration"] for data in iteration_history]
        modes = [data["mode"] for data in iteration_history]
        
        # Extract weights over iterations
        sem_weights = [data["weights"]["semantic"] for data in iteration_history]
        struct_weights = [data["weights"]["structural"] for data in iteration_history]
        bal_weights = [data["weights"]["balance"] for data in iteration_history]
        
        # Plot adaptive weights evolution
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, sem_weights, 'r-', label='Semantic Weight')
        plt.plot(iterations, struct_weights, 'g-', label='Structural Weight')
        plt.plot(iterations, bal_weights, 'b-', label='Balance Weight')
        plt.xlabel('Iteration')
        plt.ylabel('Weight Value')
        plt.title('Evolution of Adaptive Weights in GraphMaster')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('adaptive_weights_evolution.png', dpi=300, bbox_inches='tight')
        
        # Parse baseline and final reports
        baseline_data = json.loads(baseline_report)
        final_data = json.loads(final_report)
        
        # Extract label distributions
        if "LabelDistribution" in baseline_data and "LabelDistribution" in final_data:
            # Plot label distribution change
            baseline_labels = baseline_data["LabelDistribution"]
            final_labels = final_data["LabelDistribution"]
            
            labels = sorted(list(set(baseline_labels.keys()) | set(final_labels.keys())))
            baseline_counts = [int(baseline_labels.get(str(label), 0)) for label in labels]
            final_counts = [int(final_labels.get(str(label), 0)) for label in labels]
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, baseline_counts, width, label='Baseline')
            plt.bar(x + width/2, final_counts, width, label='Final')
            
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.title('Label Distribution: Baseline vs Final')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig('label_distribution_change.png', dpi=300, bbox_inches='tight')
            
        main_logger.info("[main] Visualizations generated and saved.")
    except Exception as e:
        main_logger.error(f"[main] Error generating visualizations: {e}")

def ensure_model_downloaded(model_name, cache_dir, token=None):
    """
    Checks if the specified model exists in cache_dir, 
    downloads it if not. Supports using token for gated models.
    """
    logger = logging.getLogger("main_logger")
    model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    if not os.path.exists(model_path):
        logger.info(f"[ensure_model_downloaded] Model not found in cache: {model_path}, downloading...")
        try:
            # Prepare token if provided
            token_kwargs = {"token": token} if token else {}
            
            # First try to download tokenizer
            logger.info(f"[ensure_model_downloaded] Downloading tokenizer for {model_name}...")
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **token_kwargs)
            
            # Then download model
            logger.info(f"[ensure_model_downloaded] Downloading model for {model_name}...")
            AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **token_kwargs)
            
            logger.info(f"[ensure_model_downloaded] Model downloaded successfully.")
        except Exception as e:
            logger.error(f"[ensure_model_downloaded] Error downloading model: {e}")
            
            # Special handling for authentication errors
            if "token" in str(e).lower() or "permission" in str(e).lower() or "access" in str(e).lower():
                logger.error("\n[ensure_model_downloaded] AUTHENTICATION ERROR: This model requires Hugging Face authentication.")
                logger.error("[ensure_model_downloaded] Please:")
                logger.error("  1. Request access to this model on huggingface.co")
                logger.error("  2. Run 'huggingface-cli login' in your terminal")
                logger.error("  3. Or add --hf_token YOUR_TOKEN to your command line arguments\n")
            raise
    else:
        logger.info(f"[ensure_model_downloaded] Model found in cache: {model_path}. Skipping download.")

def create_enhanced_file(data_file, enhanced_file):
    """
    Creates a copy of the data file and saves it as enhanced_file.
    """
    if not os.path.exists(data_file):
        logging.error(f"[create_enhanced_file] Error: {data_file} file does not exist!")
        return

    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def update_enhanced_data(enhanced_file, new_data_str):
    """
    Updates enhanced data by merging new enhanced data into the file.
    """
    try:
        with open(enhanced_file, 'r', encoding='utf-8') as file:
            enhanced_data = json.load(file)
    except FileNotFoundError:
        enhanced_data = []

    try:
        new_data = json.loads(new_data_str)
    except json.JSONDecodeError as e:
        logger = logging.getLogger("main_logger")
        logger.error(f"[update_enhanced_data] Error parsing new data: {e}")
        logger.error(f"[update_enhanced_data] Problematic data: {new_data_str}")
        return
    
    # Find the maximum node_id in the existing data
    try:
        last_node_id = max([int(node["node_id"]) for node in enhanced_data if isinstance(node.get("node_id"), (int, str)) and str(node["node_id"]).isdigit()], default=0)
    except Exception as e:
        logger = logging.getLogger("main_logger")
        logger.error(f"[update_enhanced_data] Error finding max node_id: {e}")
        last_node_id = 0
    
    for item in new_data:
        # Check if it's a new node with "new_node_id" or an existing node with "node_id"
        if "node_id" in item:
            node_id = item["node_id"]
            existing_node = next((node for node in enhanced_data if node["node_id"] == node_id), None)
            
            if existing_node:
                # If the node already exists, update it
                existing_node.update(item)
            else:
                # If it's a new node with a node_id, add it as is
                enhanced_data.append(item)
        elif "new_node_id" in item:
            # For new nodes with "new_node_id", assign a sequential node_id
            last_node_id += 1
            # Create a copy of the item with node_id instead of new_node_id
            new_item = item.copy()
            new_item["node_id"] = last_node_id
            # Remove the new_node_id key
            if "new_node_id" in new_item:
                del new_item["new_node_id"]
            enhanced_data.append(new_item)
        else:
            # If neither key is present, skip this item
            continue

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(enhanced_data, file, ensure_ascii=False, indent=4)

def fix_messy_json(content):
    """
    Fixes and parses messy JSON generated by LLM, ensuring extraction of pure JSON after 'here are the generated datasets:'.
    """
    # Extract content after the flag
    flag = "here are the generated datasets:"
    idx = content.lower().find(flag.lower())
    main_logger = logging.getLogger("main_logger")
    if idx == -1:
        main_logger.error("[fix_messy_json] Error: Flag 'here are the generated datasets:' not found, returning empty data.")
        return []

    content = content[idx + len(flag):].strip()

    # Only keep JSON between [ and ]
    json_pattern = r"\[\s*{.*?}\s*\]"  # Match [ {...} ] format
    match = re.search(json_pattern, content, re.DOTALL)

    if not match:
        main_logger.error("[fix_messy_json] Error: Could not match valid JSON structure, returning empty data.")
        return []

    json_str = match.group(0).strip()

    # Ensure JSON parsing succeeds
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        main_logger.error(f"[fix_messy_json] JSON parsing failed: {e}")
        main_logger.error(f"Failed JSON fragment: {repr(json_str)}")
        return []

def reformat_json_str(content):
    """
    Converts messy JSON string to standard format.
    """
    objs = fix_messy_json(content)
    return json.dumps(objs, ensure_ascii=False, indent=4)

def setup_logging():
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./log/log_main", exist_ok=True)
    os.makedirs("./log/log_result", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_logger = logging.getLogger("main_logger")
    main_logger.setLevel(logging.INFO)
    main_log_file  = f"./log/log_main/{timestamp}.log"
    main_handler = logging.FileHandler(main_log_file, mode='w', encoding="utf-8")
    main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)
    main_logger.addHandler(console_handler)  # Add terminal output

    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_log_file = f"./log/log_result/enhancement_{timestamp}.log"
    result_handler = logging.FileHandler(result_log_file, mode='w', encoding="utf-8")
    result_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    result_console_handler = logging.StreamHandler()
    result_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    result_logger.addHandler(result_handler)
    result_logger.addHandler(result_console_handler)  # Add terminal output

    return main_logger, result_logger

if __name__ == "__main__":
    main()